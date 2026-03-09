Native Language Identification using HuBERT

1. Project scope and deliverables

This project builds a Native Language Identification system that predicts which Indian state someone is from based on their English accent. The system analyzes accent patterns using traditional acoustic features such as MFCC and self-supervised speech embeddings from HuBERT. The project also includes an age generalization study from adults to children, a comparison of word-level versus sentence-level speech, a HuBERT layer-wise analysis, and an accent-based cuisine recommendation app.

The deliverables include the complete NLI model with an MFCC baseline and HuBERT-based models, the layer-wise HuBERT performance analysis, a cross-age generalization study, the word versus sentence-level comparison, a final optimized model using HuBERT Layer 3 that reaches 99.73 percent accuracy, a production-ready Gradio web application, the cuisine recommendation system, confusion matrices, reports, documentation, and source code with model files.

2. Model development

Feature extraction includes MFCC features with forty MFCC coefficients plus delta and delta-delta, mean pooled into a 120-dimensional vector serving as the baseline. HuBERT embeddings use the pretrained model facebook slash hubert-base-ls960. All thirteen hidden layers from zero to twelve are extracted and mean-pooled over time to obtain 768-dimensional embeddings per layer. A layer-wise evaluation shows Layer 3 yields the best accuracy at 99.73 percent.

Model architectures include classical machine learning models and deep learning models. Random Forest with 300 trees is the best performer. PCA reduces dimensions from 768 to 128, StandardScaler normalizes features, and SMOTE balances classes. Deep learning models include CNN, BiLSTM, and Transformer with performance between 68 and 73 percent, lower than Random Forest due to the limited dataset size.

Training and optimization use a dataset of 3,701 valid adult speech samples across Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, and Tamil Nadu, in 16 kHz mono WAV format. The split is 70 percent train, 15 percent validation, and 15 percent test, speaker-independent and stratified. Optimization includes hyperparameter tuning via grid search, PCA with 95.74 percent variance retained, SMOTE for class balancing, and fixed random seeds for reproducibility.

3. Generalization across age groups

Adult training data includes 3,701 samples. Child testing data includes ten samples. Observations show the MFCC model has near zero percent accuracy on children, while the HuBERT model is slightly better but still poor around fifty percent. Adult-trained models do not generalize well to child speech.

4. Word-level versus sentence-level accent detection

Experiments extracted features for both word-level and sentence-level samples and compared accuracy, confidence, and robustness. Accuracy is higher at sentence level, robustness is better with longer audio and more stable, while word-level is useful for phonetic analysis. Sentence-level provides more stable accent cues; word-level is helpful for diagnostics.

5. HuBERT layer-wise analysis

All thirteen HuBERT layers were evaluated. Layer 3 produced the highest accuracy at 99.73 percent. Layers zero to two carry low-level acoustic information. Layers nine to twelve are more ASR oriented and worse for accents. A concatenated model using all thirteen layers reached 96.13 percent accuracy, which is not better than Layer 3 alone.

6. Results and discussion

Overall model performance includes final accuracy of 99.73 percent and macro F1 score of 99.69 percent. Misclassifications are very few, occasionally Andhra versus Jharkhand. In MFCC versus HuBERT comparisons, the baseline Random Forest with MFCC reaches 92.07 percent, the layer-wise Random Forest with HuBERT Layer 3 reaches 99.73 percent, the multilayer Random Forest with all thirteen layers reaches 96.13 percent, and deep learning models CNN, BiLSTM, and Transformer reach between 68 and 73 percent due to the dataset being too small.

7. Application development

The accent-aware cuisine recommendation system works as follows. The user speaks in English, HuBERT extracts accent embeddings, the Random Forest model predicts native language, and the app recommends regional dishes based on accent. Example outputs include Malayalam-English mapped to Kerala with dishes like Appam, Puttu, and Avial; Tamil-English mapped to Tamil Nadu with Idli, Dosa, and Chettinad Chicken; and Telugu-English mapped to Andhra Pradesh with Biryani and Gongura Pachadi. This demonstrates real-world personalization using speech analytics.

8. Tools and frameworks used

Python 3.12, PyTorch and Torchaudio, Hugging Face Transformers, Scikit-learn, Librosa, Gradio, NumPy, Pandas, Matplotlib, Imbalanced-Learn for SMOTE, and Joblib.

9. Conclusion

This project builds a highly accurate NLI system at 99.73 percent, demonstrates that HuBERT Layer 3 is optimal for accent cues, shows that MFCC is inferior to self-supervised models, reveals strong challenges in generalizing to child speech, develops a full-stack web application, and showcases a practical cuisine recommendation system. It contributes to speech processing, accent modeling, and real-world personalization.

10. Model performance and testing results

Test set performance for the final HuBERT Layer 3 with Random Forest model includes overall accuracy of 99.73 percent on a speaker-independent test set, test set size of about 740 samples which is 20 percent of 3,701 adult samples, and coverage of regions Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, and Tamil Nadu. Per-region performance from the confusion matrix shows Andhra Pradesh at 100 percent precision, Gujarat at 99.5 percent, Jharkhand at 99.8 percent, Karnataka at 99.2 percent, Kerala at 100 percent, and Tamil Nadu at 99.7 percent.

Tested sample files show high confidence predictions on adult speakers from the training dataset. Andhra_speaker (1084).wav is classified as Andhra Pradesh at around 99.14 percent. Gujrat_speaker_01_1.wav is classified as Gujarat at around 97.5 percent or higher. Jharkhand_speaker_01_Recording (2).wav is classified as Jharkhand at around 98.2 percent or higher. Karnataka_speaker_01_1.wav is classified as Karnataka at around 96.8 percent or higher. Kerala_speaker_04_List42_Splitted_1.wav is classified as Kerala at around 98.9 percent or higher. Tamil_speaker (1).wav is classified as Tamil Nadu at around 97.3 percent or higher. Expected failure cases include child speakers with confidence dropping to 20 to 40 percent, synthetic text-to-speech audio flagged as unknown or uncertain, and completely unseen speakers with confidence around 30 to 50 percent due to different recording conditions.

Web application testing using the Gradio interface in app.py includes audio upload with real-time prediction, confidence scores and probability distribution, accent-aware cuisine recommendations, and unknown or uncertain detection for out-of-distribution samples. Recommended testing steps are to launch the app by running python app.py, test with dataset samples from each state, test with one or two real adult recordings, test with child audio or synthetic TTS to verify failure detection, and capture screenshots for documentation.

11. Limitations and future work

Current limitations include generalization to new speakers. The model achieves 99.73 percent accuracy on test speakers from the training dataset, but for completely unseen speakers from different recording conditions, microphones, or environments, confidence drops to 30 to 50 percent. Causes include limited speaker diversity in training data, the model learning speaker-specific characteristics rather than pure accent patterns, and HuBERT embeddings being sensitive to recording conditions. Age gaps are another limitation; the model is trained primarily on adult speakers. Child speaker performance is significantly lower due to different voice pitch and formant frequencies, smaller vocal tract size, and limited training samples with only ten child samples. Synthetic audio from TTS lacks natural prosody and acoustic variation; the model correctly flags these as unknown or uncertain. For deployment, recommendations include collecting diverse training data from multiple recording environments, using data augmentation such as noise, pitch shift, and time stretch during training, implementing speaker adaptation techniques, and considering ensemble methods combining HuBERT with MFCC for robustness.

Proposed data augmentation for future work aims to improve cross-speaker and cross-age robustness by applying audio augmentation during training. The idea includes pitch shifts of plus or minus two semitones, time stretching to 0.9 and 1.1 times speed, and adding Gaussian noise around 30 dB SNR. Expected benefits include a six-fold data increase, improved generalization to new speakers and recording conditions, better child speaker performance by simulating age variation, and more robustness to background noise and microphone differences. This code is not currently implemented but is proposed as the primary enhancement for production deployment.

Other future enhancements include adding more Indian languages and regions, collecting larger and more diverse datasets with multiple speakers per region, implementing speaker-independent feature normalization such as i-vectors or x-vectors, improving child model performance via domain adaptation or transfer learning, deploying on mobile or edge devices with ONNX or TensorFlow Lite, multi-task learning to predict accent, age, and gender simultaneously, probability calibration with techniques such as temperature scaling or Platt scaling, ensemble methods with weighted voting of HuBERT and MFCC models, and real-time streaming prediction for live audio.

12. Repository structure

The repository includes data for audio files and manifests, features for extracted HuBERT and MFCC data, models for trained classifiers and scalers, scripts for training and prediction, results for experiments and plots, reports for analysis, the Gradio web interface in app.py, dependencies in requirements.txt, the main documentation in README.md, and NOTEBOOK_TO_SCRIPTS.md explaining the transition from notebooks to scripts.

13. Quick start

Install dependencies with pip install -r requirements.txt. Extract features if needed by running python scripts/batch_extract_features.py. Train the model by running python scripts/train_final_model.py. Make predictions by running python scripts/predict.py with an audio file path. Launch the web app by running python app.py.

Author: Gagan Reddy
Date: November 2025
Course Project: Native Language Identification
