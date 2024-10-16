# Neuro-Net

The project was done during my fellowship at IIIT Hyderabad. The aim of the project was to create a state-of-the-art classification model for neurodegenerative diseases like ALS. Traditionally, people are called to a doctor’s office and asked to perform a certain set of oro-facial tasks which test the movement of their facial features to assess if they are healthy or not. Due to the time taken by this process and disagreement between different doctors, the need for a fast and reliable system which can aid the doctors arises. In this work, we use a recently released Toronto NeuroFace dataset to evaluate our pipeline, to the best of our knowledge this is the first publicly available dataset for neurological disease assessment.

The approach to the project is to extract the facial features of the patient from each frame, for this task a Facial Alignment Network(FAN)[1] is used, which we have implemented in PyTorch and coupled with Adaptive Wing loss[2]. After trying normal MSE loss and ASM assisted MSE loss[3] between the facial geometric points, it was found that Adaptive wing loss was able to give minimum normalised mean error on the test data. 
After extracting 68 points, multiple approaches were tried: one was to use those points and pass them through a LSTM model for classification, while another was to extract just the face using the points and then transform the perspective of the image using cv2 library. We preprocessed data in various ways to increase our dataset size by slicing each video into multiple parts containing 1-2 repetitions of the tasks and trying different architectures to achieve the best possible result. 

We also focused on C3D networks as they encompass spatial and temporal information together, also using C3D networks along with a hybrid CNN-RNN network which takes each face frame and outputs the VGG features for it and then passing those features through an LSTM layer later to be concatenated with the C3D output.

We are also used squeeze and excitation blocks so that the model can learn to pay different attention to different channels (spatial information) and different temporal information. The aim of the project was to  add clinical assistance for the classification results. Finally the approach that yielded good results was using a multimodal network with handcrafted features extracted from geometric facial points and 3D CNN
features to attain 60% accuracy on 3-way classification.

[1] Bandini A, Rezaei S, Guarin DL, Kulkarni M, Lim D, Boulos MI, Zinman L, Yunusova Y, Taati B. A new dataset for facial motion analysis in individuals with neurological disorders. IEEE Journal of Biomedical and Health Informatics. 2020 Aug 25;25(4):1111-9.
[2] Wang X, Bo L, Fuxin L. Adaptive wing loss for robust face alignment via heatmap regression. InProceedings of the IEEE/CVF international conference on computer vision 2019 (pp. 6971-6981).
[3] Fard AP, Abdollahi H, Mahoor M. ASMNet: a Lightweight Deep Neural Network for Face Alignment and Pose Estimation. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2021 (pp. 1521-1530)

![Project approach](./neuronet_approach.png)

