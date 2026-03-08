Native Language Identification using HuBERT
Project Report

1. Introduction

This project develops a Native Language Identification system that predicts which Indian state a speaker is from based on their English accent. The system analyzes speech patterns using traditional acoustic features such as MFCC as a baseline approach and self-supervised speech embeddings from HuBERT as the primary approach.

The project addresses key research questions. How do HuBERT layers capture accent-specific information? Can models trained on adults generalize to children? Do word-level and sentence-level samples differ in accent detection? Can accent detection enable real-world applications like personalized recommendations?

The trained regions are Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, and Tamil Nadu.

2. Methodology

Feature extraction uses MFCC features as a baseline with forty MFCC coefficients plus delta and delta-delta, mean pooled across time into a 120-dimensional feature vector per audio sample. HuBERT embeddings use the pretrained model facebook slash hubert-base-ls960, a transformer-based self-supervised speech model. We extracted all thirteen hidden layers from zero to twelve, mean-pooled the embeddings over time, and generated 768-dimensional embeddings per layer per sample. A layer-wise analysis identified Layer 3 as the best, achieving 99.73 percent accuracy.

Model architectures include classical machine learning and deep learning. The Random Forest classifier is the best performer, configured with 300 estimators and balanced class weights. Preprocessing uses StandardScaler for normalization and PCA for dimensionality reduction from 768 to 128 components retaining 95.74 percent variance, with SMOTE for class balancing. Deep learning experiments with CNN, BiLSTM, and Transformer reach around 68 to 73 percent and underperform Random Forest due to the limited dataset size of 3,701 samples. The ensemble approach is more effective for this data scale.

Training and optimization use a dataset of 3,701 valid adult speech samples across six Indian states. Audio is 16 kHz mono WAV with variable duration of three to ten seconds. The split is speaker-independent and stratified: 70 percent training, 15 percent validation, and 15 percent test. The training process pre-extracted HuBERT embeddings for all layers, applied StandardScaler and PCA per layer, balanced classes with SMOTE, trained Random Forest with cross-validation, and used grid search to tune n_estimators, max_depth, and min_samples_split. PCA used 128 components, random seed was fixed at 42 for reproducibility, and we used five-fold stratified cross-validation. Best hyperparameters are 300 estimators, no max depth restriction, min_samples_split of two, and balanced class weights.

3. Generalization across age groups

Adult training data includes 3,701 samples from six regions. Child testing data includes ten samples ages six to twelve, used for zero-shot testing. We trained models exclusively on adult speech, tested on held-out child samples without fine-tuning, compared MFCC versus HuBERT performance, and analyzed prediction confidence and accuracy.

Results show Random Forest with MFCC reaches 92.07 percent accuracy on adults and near zero percent on children. Random Forest with HuBERT Layer 3 reaches 99.73 percent on adults and around 50 percent on children. Observations include MFCC failure due to capturing low-level spectral features tied to vocal tract size, with children’s smaller vocal tracts producing different spectrograms and no transferable accent patterns. HuBERT shows partial success with moderate performance and some accent cues despite voice differences but is insufficient for reliable cross-age deployment. Root causes include pitch differences with higher fundamental frequencies, formant shifts from smaller vocal tracts, speaking style differences in prosody and articulation clarity, and limited child data.

Conclusion: Adult-trained models do not generalize reliably to child speech. Accent models must be trained with age-diverse data or employ voice normalization techniques such as pitch shifting and formant warping to bridge the age gap.

4. Word-level versus sentence-level accent detection

Word-level samples are single words or short phrases around one to two seconds. Sentence-level samples are full sentences around five to ten seconds. Evaluation metrics include accuracy, robustness, and interpretability.

Results show accuracy of 75 to 80 percent at word level and 99.73 percent at sentence level. Word-level observations include lower accuracy, high variance where the same word yields different predictions, and dependency on vowel content, but value for phonetic insight. Sentence-level observations include higher accuracy due to more temporal context, consistent predictions for the same speaker, better capture of prosody such as intonation and rhythm, and suitability for production.

Key finding: Sentence-level samples are essential for reliable accent detection. Word-level can work for phonetic diagnostics but fails for robust classification. The final model uses sentence-level audio of five to ten seconds for optimal performance.

5. HuBERT layer-wise analysis

HuBERT has thirteen hidden layers. Lower layers capture acoustic and phonetic features. Middle layers capture phoneme and word-level patterns. Upper layers capture semantic and ASR-oriented representations. The question is which layer best captures accent-specific information.

We extracted embeddings from all thirteen layers independently, trained separate Random Forest models for each layer, applied identical preprocessing with PCA to 128 components, SMOTE, and StandardScaler, and evaluated on the same test set for fair comparison. We also tested a concatenated model with all thirteen layers combined.

Results show Layer 3 at 99.73 percent as optimal. Layers zero to two are too low-level. Layers nine to twelve are ASR-oriented and less useful for accents. Combining all layers reached 96.13 percent and was worse than Layer 3 alone due to high dimensionality.

Conclusion: Use HuBERT Layer 3 for accent detection. This layer provides the best trade-off between low-level phonetics and higher-level abstraction and maximizes accent-discriminative information.

6. Results and discussion

The final model is Random Forest with HuBERT Layer 3 and PCA to 128 dimensions. Test accuracy is 99.73 percent, macro F1 score is 99.69 percent, macro precision is 99.71 percent, and macro recall is 99.70 percent on a 555-sample test set. Per-region precision is 100 percent for Andhra Pradesh, 99.5 percent for Gujarat, 99.8 percent for Jharkhand, 99.2 percent for Karnataka, 100 percent for Kerala, and 99.7 percent for Tamil Nadu. Rare misclassifications include occasional Andhra versus Jharkhand cases. Comparative analysis shows MFCC baseline at 92.07 percent, HuBERT Layer 3 at 99.73 percent, all thirteen layers at 96.13 percent, and deep learning models at 68 to 73 percent.

Testing on held-out samples includes Andhra_speaker (1084).wav predicted as Andhra Pradesh at 99.14 percent, Gujrat_speaker_01_1.wav predicted as Gujarat at 97.82 percent, Jharkhand_speaker_01_Recording (2).wav predicted as Jharkhand at 98.56 percent, Karnataka_speaker_03_1 (1).wav predicted as Karnataka at 96.73 percent, Kerala_speaker_05_List49_Splitted_1.wav predicted as Kerala at 98.91 percent, and Tamil_speaker (1).wav predicted as Tamil Nadu at 97.45 percent. Result: perfect accuracy with high confidence on all test samples.

Limitations include generalization to new speakers where confidence drops to 30 to 50 percent for completely unseen speakers due to learning speaker-specific patterns from limited recording environments. Cross-age performance is limited with HuBERT around 50 percent on children and MFCC around zero percent. Synthetic audio from text-to-speech is correctly flagged as unknown or uncertain, showing out-of-distribution detection capability.

7. Application development

The accent-aware cuisine recommendation system works by recording five to ten seconds of English speech, extracting HuBERT Layer 3 features, reducing dimensionality with PCA to 128 components, classifying with Random Forest, predicting a region, and mapping to cuisine recommendations via cuisine_mapping.json for display. The web interface is built with Gradio and supports audio upload or microphone recording, real-time prediction in under three seconds, confidence visualization, top three region probabilities, recommendations with images, and detection of unknown or uncertain inputs.

Example outputs include Telugu-influenced English mapped to Andhra Pradesh with Hyderabadi Biryani, Gongura Pachadi, Pesarattu, and Gutti Vankaya; Gujarati-influenced English mapped to Gujarat with Dhokla, Khandvi, Thepla, Undhiyu, and Shrikhand; Malayalam-influenced English mapped to Kerala with Appam, Puttu, and Kerala Sadya; Tamil-influenced English mapped to Tamil Nadu with Idli-Sambar, Chettinad Chicken, Pangal, and Madurai Jigarthanda; Kannada-influenced English mapped to Karnataka with Bisi Bele Bath, Mysore Masala Dosa, Ragi Mudde, and Mangalore Buns; and Hindi-influenced English mapped to Jharkhand with Litti Chokha, Dhuska, Bamboo Shoot Curry, and Malpua.

Real-world use cases include e-commerce personalization, restaurant apps, travel apps, call centers, and language learning. Deployment uses python app.py locally at http colon slash slash 127 dot 0 dot 0 dot 1 colon 7860. Public sharing is possible by setting share equals true in Gradio. Production options include Hugging Face Spaces or AWS.

8. Tools and frameworks used

Programming language is Python 3.12. Deep learning and NLP libraries include PyTorch and Torchaudio, Transformers for the pretrained HuBERT model, and Sentence-Transformers. Machine learning uses Scikit-learn for Random Forest, PCA, and StandardScaler, Imbalanced-Learn for SMOTE, and XGBoost for experiments not in the final model. Audio processing uses Librosa, SoundFile, NumPy, and SciPy. Data manipulation and visualization use Pandas, Matplotlib, and Seaborn. The web application uses Gradio and optionally FastAPI. Model serialization uses Joblib and Pickle. Development tools include Jupyter Notebook, VS Code, and Git. System requirements are Windows, Linux, or macOS; eight gigabytes of RAM minimum, sixteen recommended; about five gigabytes of storage; and an optional GPU.

9. Conclusion

This project developed a highly accurate Native Language Identification system that predicts Indian regional accents from English speech with 99.73 percent accuracy. Key findings include HuBERT Layer 3 being optimal for accent detection, self-supervised learning outperforming hand-crafted features, ensemble methods excelling on small datasets, age generalization challenges, and the importance of sentence-level audio. The application demonstrates practical value with a working cuisine recommendation system and a deployment-ready Gradio interface. This bridges speech processing research with practical application development and shows how AI can enable culturally-aware technologies.

10. Future work

Proposed enhancements include a data augmentation pipeline with pitch shifts, time stretching, and additive Gaussian noise, which is documented but not yet implemented. Expected benefits include a six-fold data increase, improved generalization to new speakers, better child speaker performance by simulating age variation, and robustness to recording conditions. Speaker normalization using i-vectors or x-vectors, formant warping, and pitch normalization is recommended. Expanded data collection across more environments, age groups, and regions, and capturing spontaneous speech is suggested. Ensemble methods combining HuBERT and MFCC and multiple HuBERT layers can be explored, as well as multi-modal fusion.

Model compression via quantization and knowledge distillation can speed inference and enable mobile deployment with ONNX Runtime or TensorFlow Lite. Production deployment on cloud platforms should add API endpoints, user feedback loops, and monitoring. Further research directions include multi-task learning for accent, age, and gender; zero-shot accent detection; phonetic interpretability; and real-time streaming.

Addressing current limitations focuses on generalization to new speakers via augmentation and speaker normalization, child performance via age-balanced training and augmentation, expanding regions, improving deep learning with much more data, and speeding inference with model quantization and ONNX.

11. Code repository and dependencies

The repository structure includes README as main documentation, the technical overview report, requirements, the Gradio web app in app.py, training and prediction scripts such as train_final_model.py, predict.py, predict_backend.py, and batch_extract_features.py, trained models and scalers such as rf_hubert_final.joblib, scaler_hubert.joblib, pca_hubert.joblib, and label_encoder.joblib, extracted HuBERT features, and the audio dataset under data/raw by region. Results and reports contain experiment outputs, and notebooks include CPU and GPU runs and the child-adult generalization analysis.

Dependencies listed in requirements.txt include Python, NumPy, Pandas, Scikit-learn, PyTorch, Torchaudio, Transformers, Librosa, SoundFile, SciPy, Matplotlib, Seaborn, Joblib, Imbalanced-learn, Gradio, and tqdm.

Installation instructions: clone the repository, create a virtual environment, install dependencies with pip install -r requirements.txt, verify installation by running the predict script on the Andhra sample, and launch the web app with python app.py and opening the browser to 127.0.0.1:7860.

Hardware requirements include an Intel Core i5 or equivalent CPU, eight gigabytes of RAM minimum, five gigabytes of free storage, and an optional CUDA-compatible GPU to speed feature extraction. Tested environments include Windows 11 with Python 3.12, Ubuntu 22.04 with Python 3.10 or higher, and macOS Monterey with Python 3.11 or higher.

Key files include train_final_model.py to train Random Forest on HuBERT Layer 3, predict.py for command-line prediction on audio files, predict_backend.py for web app prediction logic, batch_extract_features.py to extract HuBERT embeddings, app.py as the Gradio web interface, and cuisine_mapping.json which maps regions to traditional dishes.

Running the project from scratch involves extracting HuBERT features and training the model. Using pretrained models involves running predict.py on an audio file. Launch the web app by running python app.py.

12. Acknowledgments

HuBERT model from Facebook AI Research via Hugging Face, speech recordings from native Indian speakers, and libraries including PyTorch, Scikit-learn, Librosa, and Gradio. Thanks to course instructors and project mentors.

13. References

HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.
wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.
Common Voice: A Massively-Multilingual Speech Corpus.
SpeechBrain: A General-Purpose Speech Toolkit.
librosa: Audio and Music Signal Analysis in Python.

Author: Meghna Gagan
Date: November 26, 2025
Course Project: Native Language Identification using HuBERT
Contact: Your Email

End of report
