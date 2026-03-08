from pathlib import Path
import numpy as np
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persistent_hubert import extract_live_features, init_hubert
import soundfile as sf
import tempfile
import uuid
from predict import load_models  # your existing loader

BEST_LAYER = 3

# Optional: load pairwise verifier + calibration here
AJ_VERIFIER_PATH = Path("models/andhra_jharkhand_verifier.joblib")
AJ_SCALER_PATH = Path("models/andhra_jharkhand_scaler.joblib")
AJ_PCA_PATH = Path("models/andhra_jharkhand_pca.joblib")
AJ_INFO_PATH = Path("models/andhra_jharkhand_info.json")

_hubert_inited = False
_model_bundle = None
_aj_bundle = None

def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = load_models()  # (model, scaler, pca, le)
    return _model_bundle

def ensure_hubert(device="cpu"):
    global _hubert_inited
    if not _hubert_inited:
        init_hubert(device)
        _hubert_inited = True

def get_aj_bundle():
    global _aj_bundle
    if _aj_bundle is None and AJ_VERIFIER_PATH.exists() and AJ_SCALER_PATH.exists() and AJ_INFO_PATH.exists():
        import json
        aj_clf = joblib.load(AJ_VERIFIER_PATH)
        aj_scaler = joblib.load(AJ_SCALER_PATH)
        aj_pca = joblib.load(AJ_PCA_PATH) if AJ_PCA_PATH.exists() else None
        with open(AJ_INFO_PATH, 'r', encoding='utf-8') as f:
            aj_info = json.load(f)
        _aj_bundle = (aj_clf, aj_scaler, aj_pca, aj_info)
    return _aj_bundle

def predict_from_path(audio_path: str, device: str = "cpu"):
    """
    Unified backend used by both Gradio and Streamlit.
    Returns the same dict structure as your current predict_native_language().
    """
    model, scaler, pca, le = get_model_bundle()
    audio_path = Path(audio_path)
    stem = audio_path.stem

    # 1) cached-first lookup (only works when user browses actual file from data/raw/)
    # Note: UI uploads create temp files like "name.wav", so cache typically won't hit
    cached_path = Path("features/hubert") / f"{stem}.npz"
    
    segment_votes = None
    used_segmentation = False
    if cached_path.exists():
        data = np.load(cached_path, allow_pickle=True)
        pooled = data["pooled"]
        print(f"  [CACHE HIT] Loaded features from: {cached_path.name}")
    else:
        # 2) live extraction (+ optional segmentation) + calibration
        ensure_hubert(device)
        # Inspect duration for segmentation decision
        try:
            audio_raw, sr = sf.read(str(audio_path))
            duration = len(audio_raw) / sr if sr else 0.0
        except Exception:
            audio_raw, sr, duration = None, None, 0.0

        # Use segment voting for clips >= 6s
        if audio_raw is not None and duration >= 6.0:
            seg_seconds = 2.5
            seg_samples = int(seg_seconds * sr)
            per_segment_probs = []
            per_segment_labels = []
            # Non-overlapping segments
            for start in range(0, len(audio_raw) - seg_samples + 1, seg_samples):
                segment = audio_raw[start:start + seg_samples]
                if len(segment) < seg_samples:
                    break
                tmp_path = Path(tempfile.gettempdir()) / f"nli_seg_{uuid.uuid4().hex}.wav"
                try:
                    sf.write(str(tmp_path), segment, sr)
                    seg_pooled = extract_live_features(str(tmp_path))  # (13,768)
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                # Layer normalization ([:13] safeguard) + mean over layers
                if seg_pooled.shape[0] > 13:
                    seg_pooled = seg_pooled[:13]
                seg_vec = seg_pooled.mean(axis=0).reshape(1, -1)
                seg_scaled = scaler.transform(seg_vec)
                seg_pca = pca.transform(seg_scaled) if pca is not None else seg_scaled
                seg_probs = model.predict_proba(seg_pca)[0]
                per_segment_probs.append(seg_probs)
                per_segment_labels.append(int(np.argmax(seg_probs)))
            if per_segment_probs:
                used_segmentation = True
                avg_probs = np.mean(per_segment_probs, axis=0)
                segment_votes = {
                    "segments": len(per_segment_probs),
                    "majority_label": le.inverse_transform([int(np.bincount(per_segment_labels).argmax())])[0],
                    "vote_distribution": {le.classes_[i]: int(np.sum(np.array(per_segment_labels)==i)) for i in range(len(le.classes_))},
                }
            # If segmentation provided at least 2 segments we skip pooled feature path entirely.
            if not used_segmentation:
                pooled = extract_live_features(str(audio_path))
            else:
                pooled = None  # signal downstream to use avg_probs directly
        else:
            pooled = extract_live_features(str(audio_path))

        calib_path = Path("models/live_calibration.npz")
        if calib_path.exists():
            try:
                calib = np.load(calib_path)
                layer_vec = pooled[BEST_LAYER]
                mean_cached = calib["mean_cached"]
                std_cached = calib["std_cached"]
                mean_live = calib["mean_live"]
                std_live = calib["std_live"]
                mapped = ((layer_vec - mean_live) / std_live) * std_cached + mean_cached
                pooled[BEST_LAYER] = mapped
            except Exception:
                pass

    # 3) Normalize layer count for consistency, then average → scaler → PCA → RF
    # HuBERT-base returns 13 hidden states (1 embedding + 12 layers) for live extraction.
    # Some cached files contain 26 rows (legacy double-stack). To match training and inference,
    # use only the first 13 rows when more are present.
    if used_segmentation and pooled is None:
        # Direct probability aggregation path
        probs = avg_probs
    else:
        if pooled.ndim == 2 and pooled.shape[0] > 13:
            pooled = pooled[:13]
        layer_vec = pooled.mean(axis=0).reshape(1, -1)
        layer_scaled = scaler.transform(layer_vec)
        layer_pca = pca.transform(layer_scaled) if pca is not None else layer_scaled
        probs = model.predict_proba(layer_pca)[0]

    # Load open-set class stats if available
    stats_path = Path("models/class_stats.npz")
    mahal_distance = None
    if stats_path.exists():
        try:
            stats = np.load(stats_path, allow_pickle=True)
            means = stats['means'].item()
            inv_covs = stats['inv_covs'].item()
            distance_threshold = float(stats['distance_threshold'])
            # Compute per-class Mahalanobis; take min
            d_min = None
            for cls_idx, mu in means.items():
                inv_cov = inv_covs.get(cls_idx)
                diff = (layer_pca - mu)
                d = float(diff @ inv_cov @ diff.T)
                if d_min is None or d < d_min:
                    d_min = d
            mahal_distance = d_min
        except Exception:
            mahal_distance = None

    # (probs already defined above for both segmentation and single-pass paths)
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    # 4) Build top-3
    top_indices = np.argsort(probs)[::-1][:3]
    top_labels = le.inverse_transform(top_indices)
    top3 = [(lbl, float(probs[i])) for lbl, i in zip(top_labels, top_indices)]

    # Low-confidence logic + unknown threshold (more relaxed for broad usability)
    margin = float(probs[top_indices[0]] - probs[top_indices[1]]) if len(top_indices) > 1 else 1.0
    low_confidence = confidence < 0.50 or margin < 0.10
    unknown_threshold = (confidence < 0.25 or margin < 0.06)

    # Remove strict confusion-cluster unknowning to stay flexible for users
    # (We still report low_confidence and margin for transparency.)

    # If segmentation majority disagrees with top1 and margin small → unknown
    if used_segmentation and segment_votes:
        majority = segment_votes.get("majority_label")
        if majority != top3[0][0] and margin < 0.12:
            unknown_threshold = True
    if mahal_distance is not None:
        # Use distance threshold if available to enhance unknown detection
        # Only flag as unknown if BOTH distance is high AND confidence is very low
        if mahal_distance > distance_threshold * 1.3 and confidence < 0.50:
            unknown_threshold = True

    # Pairwise Andhra/Jharkhand override
    aj_bundle = get_aj_bundle()
    if aj_bundle is not None:
        aj_clf, aj_scaler, aj_pca, aj_info = aj_bundle
        aj_classes = set(aj_info.get('classes', ['andhra_pradesh','jharkhand']))
        top1_label, top1_prob = top3[0]
        top2_label, top2_prob = top3[1] if len(top3) > 1 else (None, 0.0)
        ambiguous_aj = (top1_label in aj_classes and top2_label in aj_classes)
        if ambiguous_aj and (low_confidence or confidence < 0.50):
            multi_vec = pooled.reshape(1, -1)
            multi_vec_s = aj_scaler.transform(multi_vec)
            if aj_pca is not None:
                multi_vec_p = aj_pca.transform(multi_vec_s)
            else:
                multi_vec_p = multi_vec_s
            pair_probs = aj_clf.predict_proba(multi_vec_p)[0]
            mapping = aj_info.get('mapping', { 'andhra_pradesh': 0, 'jharkhand': 1 })
            inv_map = {v:k for k,v in mapping.items()}
            pair_pred_idx = int(np.argmax(pair_probs))
            pair_pred_label = inv_map.get(pair_pred_idx, top1_label)
            pair_pred_conf = float(np.max(pair_probs))
            if pair_pred_conf > 0.55 and pair_pred_label != pred_label:
                pred_label = pair_pred_label
                confidence = pair_pred_conf
                # Update top3 and all_probabilities
                all_probs = {cls: float(p) for cls, p in zip(le.classes_, probs)}
                all_probs[pair_pred_label] = max(all_probs.get(pair_pred_label, 0.0), pair_pred_conf)
                top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    # If below unknown threshold, mark as uncertain
    if unknown_threshold:
        output_label = "unknown/uncertain"
    else:
        output_label = pred_label

    result = {
        "predicted_label": output_label,
        "base_label": pred_label,
        "confidence": confidence,
        "top_3": top3,
        "all_probabilities": {cls: float(p) for cls, p in zip(le.classes_, probs)},
        "low_confidence": low_confidence,
        "margin": margin,
        "unknown": unknown_threshold,
        "mahal_distance": mahal_distance,
        "used_segmentation": used_segmentation,
        "segment_votes": segment_votes,
    }
    return result

def log_edge_case(audio_path: str, result: dict):
    """Log low-confidence or unknown predictions to CSV for later analysis."""
    if not (result.get("low_confidence") or result.get("unknown")):
        return
    import csv
    import datetime
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "prediction_review.csv"
    write_header = not log_file.exists()
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "audio": str(audio_path),
        "predicted_label": result.get("predicted_label"),
        "base_label": result.get("base_label"),
        "confidence": result.get("confidence"),
        "margin": result.get("margin"),
        "unknown": result.get("unknown"),
        "mahal_distance": result.get("mahal_distance"),
        "top3": ";".join([f"{lbl}:{prob:.4f}" for lbl, prob in result.get("top_3", [])])
    }
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
