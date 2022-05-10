import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from PIL import Image
from helper import pred_and_plot, load_and_prep_image
import sys
from details import fish_details

# Weights
fish_classifier = tf.keras.models.load_model(r"Weights/fish_classification.h5")
plant_disease_classifier = tf.keras.models.load_model(r"Weights/plant_disease_augmented_dataset.h5")
with open(r"Weights/fish_reco.sav", "rb") as input_file:
    fish_reco = pickle.load(input_file)

PAGE_CONFIG = {"page_title": "Aquaponics Hanbook", "page_icon": r'Website Images/fish.png', "layout": "wide", "initial_sidebar_state": "auto"}
st.set_page_config(**PAGE_CONFIG)

# Functions
@st.cache
def load_image(image_file, image_size=(224, 224)):
    img = Image.open(image_file)
    resizedImg = img.resize(image_size, Image.Resampling.LANCZOS)
    return resizedImg


def main():

    st.title(f"Aquaponics Handbook 📘")
    st.markdown("<hr>", True)
    menu = ['Home', 'Fish Classifier', 'Fish Recommendation (BETA)', 'Plant Disease Classifier', 'About']
    peripheral_active = False
    with st.sidebar:
        st.title("Aquaponics Tools")
        choice = st.selectbox("Menu", menu)
        st.title("Peripherals")
        with st.expander("Reseach Paper"):
            intro = st.button("Introduction")
            method = st.button("Methodology")
            rad = st.button("Results and Discussions")
            conc = st.button("Conclusion")
            ref = st.button("References")
        with st.expander("Datasets"):
            st.button("Fish Classification Dataset")
            st.button("Fish Pond Dataset")
            st.button("Plant Village Dataset")

        st.image(load_image('Website Images/aqua-logo.png', (200, 300)))

    #################################################### RESEARCH PAPER ##################################################
    if intro:
        peripheral_active = True

        st.header("Introduction")
        st.write("""
        Aquaponics is a typical bio-integrated method of production that links the cyclic nature of aquaculture with hydroponics, which shares a sustainable agricultural system [1]. It merges fish cultivation(aquaculture) and plant farming(hydroponics), without the presence of soil at the base of the setup. The mechanism works by combining these two agricultural sciences to sustain the recirculating system by natural biological cycles (nitrification) to supply nitrogen and reduce water inputs together with the non-renewable fertilizers [2]. Several systems, techniques, design, scales and nomenclature within aquaponics have been developed throughout the years and it has already been commercialized and used in many scientific studies [3].
        """)

        st.write("""
        Producing plants hydroponically and farming fish using aqua¬culture requires special methodologies to make sure that the respective systems are implemented and managed properly [2]. The complexity of each system models is proportional to the parameters being considered and monitored across varying scales and purpose of implementation. Problems may occur when these parameters do not align with the acceptable thresholds that were initially set or external factors like the presence of disease-causing agents for the crops and unwanted chemicals in the waters are seen upon the actual implementation. In the real world, it is very difficult for fish farmers to recommend the perfect fish species for aquaculture in a specific aquaponics system [4]. Land farmers may also be unfamiliar with the merger since growing crops through a land-based system is significantly different compared to an aquaponics environment. Whatever the case may be, investing into the production of an aquaponic system whether home-based or commercialized, is an expensive and risky venture if not equipped with the necessary guides and practices to not just make it work properly, but also sustainably maintaining its mechanisms.
        """)

        st.write("""
        Within the emergence of IoT based smart systems and the advancements in Artificial Intelligence, more innovative approaches would be developed to further advance the field of aquaponics and minimize the trade-offs enabling it to become a household or commercial venture. In this new era of science and technology, computer-aided diagnosis (CAD) systems are being used and developed for agricultural sciences along with the applications of Machine Learning and automation [5]. There are many applications of AI (Deep Learning and Machine Learning) in the aquaculture and hydroponics field and several journal articles have been published to provide empirical methods and sound practices that are based on gathered data and insights made through data mining [4]–[8]. Although both these fields share many common intersection with aquaponics, the implementation and scale are relatively different. In a general view, aquaponics encompasses both these fields and merges them to create a feedback and two player system that enables survival of various living agents within the model [9].
        """)

        st.write("""
        Proper recommendations and guidelines for aquaponic farming are still lacking and requires more research investments focusing on possible production systems (type of fish × type of plant × densities × filtration system × hydroponic system × aquaculture system) and different varieties of crops [2]. There are studies that integrate Internet of Things (IoT) devices within their aquaponic systems to gather data and make sense about the patterns that occur [10]. This intersection is commonly referred as “smart aquaponics” where data and information are core elements.  However, measured data are oftentimes affected by complicated environmental factors that are usually nonlinear and various, which increases the complexity and difficulty of maintaining an accurate control system [11]. The massive amount of data in smart aquaponics imposes a variety of challenges, such as multiple sources, data inconsistencies, data volatilities, and structural complexities. To find a way to convert this data into insights and imperatives, various sophisticated Machine Learning(ML) algorithms are being used. For image data, for example, Deep Learning can be utilized because it involves multilevel data representations, from low to high levels, in which high-level features are built on the low-level features and carry rich semantic information that can be used to recognize and detect targets or objects in the image [11]. Deep Learning is commonly used together with other Machine learning models to makes sense of unstructured data like images or text. There are also other Machine Learning algorithms that can be used for structured or tabular datasets that can be utilized for regression or classification problems without the use of sophisticated Deep Learning techniques.  By combining IoT, Data Mining, Artificial Intelligence and Data Science, it is possible to create a framework or standard that would aid beginning agriculturists or the common people to implement their own aquaponics system.
        """)

        st.write("""
        Transforming data from previously acquired analytics or creating an image classification deep learning algorithm are just some of the possibilities that can be implemented through Artificial Intelligence that would help non-experts of the field have a data-driven decisions for implementing their own aquaponics system. As of the writing of this paper, there are only handful of AI-based papers that focuses solely on aquaponics and to the best of my knowledge, this is the first paper to provide a handbook for aquaponics covering several preliminaries using both Machine Learning and Deep Learning from the gathered datasets.
        """)

    elif method:
        peripheral_active = True
        st.header("Methodology")

        st.write("""
        The underlying step-by-step process followed when building the machine learning models are presented on Appendix 1. The processes within the workflows were thoroughly discussed and the models were specifically created for each respective feature present at the Aquaponics Handbook website.
        """)

        st.subheader("The dataset")
        st.write("The varying datasets were all gathered from Kaggle.com. The largescale fish dataset was adapted from the fish segmentation and classification study by Ulucan, et al. The dataset is composed of raw fish images separated through nine different classes namely: Black Sea Sprat, Gilt Head Beam, Horse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet, and Trout. Images were collected via 2 different cameras, Kodak Easyshare Z650 and Samsung ST60. Therefore, the resolution of the images are 2832 x 2128, 1024 x 768, respectively [12]. For each class, there are 1000 augmented images and their pair-wise augmented ground truths. However, this paper only considered the raw 1000 augmented images and completely disregarded the others. The sample images from randomly chosen classes are present in Figure 1.")
        st.image(r"Paper Images\sample fish images.png")

        st.write("The image dataset for plant disease classification was taken from a publicly known dataset called Plant Village from Pennsylvania State University that is available in many machine learning repositories. In this data-set, 38 different classes of plant leaf and background images are available. These classes include 13 different plant types some with many available images of different plant diseases. In this study, the proposed deep learning model utilized the augmented version of the dataset by Geetharamani and Arun Pandian in their paper “Identification of plant leaf diseases using a nine-layer deep convolutional neural network” [13]. Overall, the data-set contained 54, 305 images. The sample images from randomly chosen classes are presented in Figure 2. ")
        st.image(r"Paper Images\sample plants images.png")

        st.subheader("Experimental Setup")
        st.write("The Image classification features of the aquaponic handbook website utilized the pre-trained Deep Convolutional Neural Network(CNN) model called MobileNetV2 as base model for transfer-learning [14], [15]. This was implemented through the TensorFlow, Keras, and pandas libraries which use the Python programming language. The source code was written on Google Collaboratory which offers a free Graphics Processing Unit(GPU) that makes training and testing the models faster. The trained weights were then downloaded locally and integrated to a Streamlit backend which was used to create the website. The overall web application was then uploaded at GitHub and hosted at Streamlit sharing.")

        st.subheader("Image Classifcation")
        st.write("The schematic in Figure 3 depicts a potential view for the image classification and analysis of the available features in the aquaponics handbook website. Initially, plant leaf disease and market fish images are collected and classified into several categories. The image file paths were then arranged with its corresponding labels through a pandas Data frame to structure the data into two columns(file paths and labels) with the individual images as its rows. The data was then split into three categories namely: training, validation, and testing. Each category has a target size of 224 x 224, color mode is ‘rgb’, class mode is categorical, and the batch size is 32. The individual images were also preprocessed by scaling them to range from 0 to 1. Random flip, random rotation, random zoom, height adjustment, and width adjustment are some of the data augmentation techniques used. By using data augmentation methods, new sample photos are created from available photos to enhance and prepare the dataset [8]. In this case, only the images from the plant village dataset undergone data augmentation preprocessing. The photos are then used as input to the suggested approach for training the model in the following stage. After undergoing through transfer learning, the data from the base model is then connected to two Dense layers with 256 hidden fully connected(FC) layers and the activation function used was Rectified Linear Unit (ReLU). In the fish classification feature, only 128 fully connected layers are used within the two Dense layers.  Both the FC layers are separated by a Dropout function with a value of 0.2 respectively [16]. The dropout function was only applied to the images in the plant village dataset since there is a tendency for the model to overfit in the training data. The newly trained architectural model will be used to anticipate previously unseen images. In this case, the model was evaluated on the test data from the previous split. Eventually the findings of the classifications are achieved.")
        st.image(r"Paper Images\Schematic Diagram.png")

        st.subheader("Transfer Learning")
        st.write("The Deep Learning model’s optimization and training is a computationally intensive and time-consuming operation. As mentioned earlier, a powerful graphics processing unit(GPU) is required for training the model, as well as large amounts of data. However, transfer learning, which is deployed in deep learning, solves these problems [8]. The pre-trained Convolutional Neural Network (CNN) used in transfer learning is optimized for one task and transfers knowledge to different modes [8], [17]. The images from the gathered datasets were compromised of different file sizes. Due to this, it was resized to a size of 224 x 224 with three channels to cater its ‘rgb’ type. The pre-trained CNN used is MobileNetV2 was used to find patterns within the input images and it’s corresponding final layers have to be connected to a dense layer. The final layers before the softmax is a 11 x 11 Dense layer for the Fish classification, and a 38 x 38 Dense for the plant disease classification. The basic picture preparation is necessary for the transfer learning considerations with the data augmented images. Figure 4 shows the whole ")
        st.image(r"Paper Images\Neural network architecture.png")

        st.subheader("Model Evaluation")
        st.write("The test dataset gathered from the splitting of the dataset during the data preparation stage was used to evaluate the trained model. Classification metrics from scikit-learn library were also used. This includes the classification reports and confusion matrix. Throughout the training of the model, the training accuracy, training loss, validation accuracy, and validation loss were also tracked to determine if the model is overfitting or underfitting.")

        st.subheader("Aquaponics Handbook Website")
        st.write("The saved weights from the trained models were downloaded locally. The image classification features were made through the Streamlit library that use the Python programming language. The finished website was then uploaded to GitHub pages together with the images, weights, and helper python files. ")

    elif rad:
        peripheral_active = True
        st.header("Results and Discussions")

        st.subheader("Metrics")
        st.write("The metrics used to evaluate how well the model performs are precision, recall, F1 score and the confusion matrix. Higher precision leads to less false positives, higher recall leads to less false negatives, f1-score is the combination of precision and recall, usually a good overall metric for classification model. When comparing predictions to truth labels, a confusion matrix can be used to see where the model gets confused.")

        st.markdown("<h5>Precision (P)</h5>", True)
        st.write("The fraction of true positives (TP, correct predictions) from the total amount of relevant results, i.e., the sum of TP and false positives (FP). For multi-class classification problems, P is averaged among the classes [7]. ")

        st.markdown("<h5>Recall (R)</h5>", True)
        st.write("The fraction of TP from the total amount of TP and false negatives (FN). For multi-class classification problems, R gets averaged among all the classes [7]. ")

        st.markdown("<h5>F1 Score (F1)</h5>", True)
        st.write("The harmonic mean of precision and recall. For multi-class classification problems, F1 gets averaged among all the classes [7]. It is mentioned as F-measure in [18].")

        st.subheader("Fish Classification Evaluation")
        st.write("The images from the fish classification was trained using the proposed Deep Convolutional Neural Network. The model scored a test accuracy of 99.78% and a test loss of 0.00722. The timeline for the accuracy and loss of the training and validation datasets are presented on Figure 5 and Figure 6 respectively. The corresponding classification reports are presented on Table 1 and the confusion matrix on Figure 7. Sample predictions on the test data are presented on Figure 8.")
        st.image(r"Paper Images\Fish classification accuracy.png")
        st.image(r"Paper Images\Fish Classification loss curves.png")
        df_fish = pd.read_csv(r"Paper Tables\Fish classification reports.txt")
        st.table(df_fish)
        st.image(r"Paper Images\fish classification confusion matrix.png")

        st.subheader("Plant Disease Classification Evaluation")
        st.write("The images from the plant disease classification was trained using the proposed Deep Convolutional Neural Network. The model scored a test accuracy of 96.35% and a test loss of 0.11303. The timeline for the accuracy and loss of the training and validation datasets are presented on Figure 9 and Figure 10 respectively. The corresponding classification reports are presented on Table 2 and the confusion matrix on Figure 11. Sample predictions on the test data are presented on Figure 12.")
        st.image(r"Paper Images\plant disease accuracy.png")
        st.image(r"Paper Images\Plant disease loss curves.png")
        df_plants = pd.read_csv(r"Paper Tables\plant disease classification reports.txt")
        st.table(df_plants)
        st.image(r"Paper Images\plant disease confusion matrix.png")

        st.subheader("Website Deployment")
        st.write("The website was deployed through the Streamlit sharing feature associated with GitHub. Figure 13 shows the interface of the website.")
        st.image(r"Paper Images\website interface.JPG")

    elif conc:
        peripheral_active = True
        st.header("Conclusion")
        st.write("In this paper, two datasets were collected, prepared, analyzed, and implemented as a feature for an Aquaponics helper website. These datasets are fish images and plant disease images taken from previous similar studies available in public repositories. The techniques of data augmentation, dataset pre-processing, training, and testing are applied to the convolutional neural network-based MobileNetV2 model. The proposed model is built and tested to improve the performance measured and compared it throughout different parameters. The evaluation metrics like precision, recall, and f1-score were utilized to evaluate the model as the hyperparameters are adjusted to determine the best model for the image classification. After the training and evaluation, two classifier weights were eventually achieved. The first model which was trained with the fish images achieved a test accuracy of 99.78% and loss of 0.00722. On the other hand, the second model which was trained with the plant disease images achieved a test accuracy of 96.35% and a loss of 0.11303. Always improving the performance of our models for fish and plant disease classification and analysis is a critical step, but our model achieved a satisfactory performance, which hopefully will support aquaponics practitioners. The major focus of this research is to provide a helper tool that would help non-experts in the field aquaponics to have a guide for data driven decisions about creating and maintaining an aquaponic system. The collection and preparation of genuine datasets and it applications to aquaponics to add more features to the website is a future target. Using more sophisticated deep learning models and gathering more datasets for better classification and analysis are anticipated. My work encourages and stimulates aquaponics practitioners and hobbyist alike, which ultimately helps raise their respective incomes and promotes a bio-friendly way of agriculture that could be implemented by anyone.")

    elif ref:
        peripheral_active = True
        st.header("References")
        st.text("""

        [1]	P. G. Panigrahi, S. Panda, and S. N. Padhi, “Aquaponics: An innovative approach of symbiotic farming,” IJB, vol. 5, no. 09, p. 4808, Aug. 2016, doi: 10.21746/ijbio.2016.09.005.
        [2]	R. V. Tyson, “Aquaponics—Sustainable Vegetable and Fish Co-Production,” p. 5.
        [3]	H. W. Palm et al., “Towards commercial aquaponics: a review of systems, designs, scales and nomenclature,” Aquacult Int, vol. 26, no. 3, pp. 813–842, Jun. 2018, doi: 10.1007/s10499-018-0249-z.
        [4]	Md. M. Islam, M. A. Kashem, and J. Uddin, “Fish survival prediction in an aquatic environment using random forest model,” IJ-AI, vol. 10, no. 3, p. 614, Sep. 2021, doi: 10.11591/ijai.v10.i3.pp614-622.
        [5]	V. K. Shrivastava, M. K. Pradhan, S. Minz, and M. P. Thakur, “RICE PLANT DISEASE CLASSIFICATION USING TRANSFER LEARNING OF DEEP CONVOLUTION NEURAL NETWORK,” Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., vol. XLII-3/W6, pp. 631–635, Jul. 2019, doi: 10.5194/isprs-archives-XLII-3-W6-631-2019.
        [6]	S. Dilmi and M. Ladjal, “A novel approach for water quality classification based on the integration of deep learning and feature extraction techniques,” Chemometrics and Intelligent Laboratory Systems, vol. 214, p. 104329, Jul. 2021, doi: 10.1016/j.chemolab.2021.104329.
        [7]	A. Kamilaris and F. X. Prenafeta-Boldú, “Deep learning in agriculture: A survey,” Computers and Electronics in Agriculture, vol. 147, pp. 70–90, Apr. 2018, doi: 10.1016/j.compag.2018.02.016.
        [8]	A. S. Paymode and V. B. Malode, “Transfer Learning for Multi-Crop Leaf Disease Image Classification using Convolutional Neural Network VGG,” Artificial Intelligence in Agriculture, vol. 6, pp. 23–33, 2022, doi: 10.1016/j.aiia.2021.12.002.
        [9]	D. Karimanzira, C. Na, M. Hong, and Y. Wei, “Intelligent Information Management in Aquaponics to Increase Mutual Benefits,” IIM, vol. 13, no. 01, pp. 50–69, 2021, doi: 10.4236/iim.2021.131003.
        [10]	C. S. Arvind, R. Jyothi, K. Kaushal, G. Girish, R. Saurav, and G. Chetankumar, “Edge Computing Based Smart Aquaponics Monitoring System Using Deep Learning in IoT Environment,” in 2020 IEEE Symposium Series on Computational Intelligence (SSCI), Canberra, ACT, Australia, Dec. 2020, pp. 1485–1491. doi: 10.1109/SSCI47803.2020.9308395.
        [11]	X. Yang, S. Zhang, J. Liu, Q. Gao, S. Dong, and C. Zhou, “Deep learning for smart fish farming: applications, opportunities and challenges,” Rev. Aquacult., vol. 13, no. 1, pp. 66–90, Jan. 2021, doi: 10.1111/raq.12464.
        [12]	O. Ulucan, D. Karakaya, and M. Turkan, “A Large-Scale Dataset for Fish Segmentation and Classification,” in 2020 Innovations in Intelligent Systems and Applications Conference (ASYU), Istanbul, Turkey, Oct. 2020, pp. 1–5. doi: 10.1109/ASYU50717.2020.9259867.
        [13]	G. Geetharamani and J. Arun Pandian, “Identification of plant leaf diseases using a nine-layer deep convolutional neural network,” Computers & Electrical Engineering, vol. 76, pp. 323–338, Jun. 2019, doi: 10.1016/j.compeleceng.2019.04.011.
        [14]	Y. LeCun, Y. Bengio, and T. B. Laboratories, “Convolutional Networks for Images, Speech, and Time-Series,” p. 15.
        [15]	M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” arXiv:1801.04381 [cs], Mar. 2019, Accessed: May 07, 2022. [Online]. Available: http://arxiv.org/abs/1801.04381
        [16]	N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overﬁtting,” p. 30.
        [17]	P. Nevavuori, N. Narra, and T. Lipping, “Crop yield prediction with deep convolutional neural networks,” Computers and Electronics in Agriculture, vol. 163, p. 104859, Aug. 2019, doi: 10.1016/j.compag.2019.104859.
        [18]	D. H. T. Minh et al., “Deep Recurrent Neural Networks for mapping winter vegetation quality coverage via multi-temporal SAR Sentinel-1,” arXiv:1708.03694 [cs], Aug. 2017, Accessed: May 08, 2022. [Online]. Available: http://arxiv.org/abs/1708.03694

        """)


    ####################################################### MENU ##########################################################

    if choice == "Home" and peripheral_active is False:
        st.subheader("Home")
        st.write("In recent times, the use of artificial intelligence(AI) in aquaculture and hydroponics is rapidly becoming the norm as more data is being acquired and generated. The advancements in Internet of Things(IoT) and Machine Learning amplified the data gathering process and extended the development of new approaches in image classification and pattern recognition. However, the field of aquaponics, which encompasses both mentioned agricultural sciences, also have the same potential especially when portrayed through an economic or scientific perspective. The main objective that this research is trying to achieve is to provide a helper tool for non-practitioners of aquaponics through a data-driven web application that uses the concepts of deep learning as a subset of AI. Two image datasets were acquired composing of fish and plant diseases which will be used for image classification. Fish and plants play a core element in every aquaponic system, and it is important to have a tool that would ensure accurate decisions when handling both these components. The MobileNetV2 Deep Convolutional Neural Network was used as  a transfer learning base model for the image classification features of the website as it is fits in many platforms and lightweight for mobile devices. The performance measure parameters, i.e., precision, recall, F1-score, and test accuracy were also calculated and monitored. The designed model achieved a 99.78% test accuracy for the fish classification feature and a 96.35% test accuracy for the plant-disease classification feature. The proposed research directly supports the intersection of artificial intelligence and aquaponics as it is beneficial in terms of biodiversity and economics. ")
        st.image(r"Website Images\Home image 1.png")
        st.write("Aquaponics is a food production system that couples aquaculture with the hydroponics whereby the nutrient-rich aquaculture water is fed to hydroponically-grown plants, where nitrifying bacteria convert ammonia into nitrates.")
        st.image(r"Website Images\Home image 2.jpg")

        st.markdown("<hr>", True)
        st.subheader("What is Aquaponics?")
        st.write("""
        Many definitions of aquaponics recognize the ‘ponics’ part of this word for hydroponics which is growing plants in water with a soil-less media.  Hydroponics is its own growing method with pros and cons (discussed later).

        Literally speaking, Aquaponics is putting fish to work. It just so happens that the work those fish do (eating and producing waste), is the perfect fertilizer for growing plants. And man,  fish can grow a lot of plants when they get to work!

        One of the coolest things about Aquaponics is that it mimics a natural ecosystem. Aquaponics represents the relationship between water, aquatic life, bacteria, nutrient dynamics, and plants which grow together in waterways all over the world. Taking cues from nature, aquaponics harnesses the power of bio-integrating these individual components:  Exchanging the waste by-product from the fish as a food for the bacteria, to be converted into a perfect fertilizer for the plants, to return the water in a clean and safe form to the fish. Just like mother nature does in every aquatic ecosystem.
        """)

        st.success("Aquaponics uses the best of all the growing techniques, utilizing the waste of one element to benefit another mimicking a natural ecosystem.  It’s a game changer")
        st.write("""
        - Waist-high aquaponic gardening eliminates weeds, back strain, and small animal access to your garden.
        - Aquaponics relies on the recycling of nutrient-rich water continuously. In aquaponics, there is no toxic run-off from either hydroponics or aquaculture.
        - Aquaponics uses 1/10th of the water of soil-based gardening and even less water than hydroponics or recirculating aquaculture.
        - No harmful petrochemicals, pesticides or herbicides can be used. It’s a natural ecosystem.
        - Gardening chores are cut down dramatically or eliminated. The aquaponics grower is able to focus on the enjoyable tasks of feeding the fish and tending to and harvesting the plants.
        - Aquaponic systems can be put anywhere, use them outside, in a greenhouse, in your basement, or in your living room. By using grow-lighting, and space can become a productive garden.
        - Aquaponic systems are scalable. They can fit most sizes and budgets, from small countertop herb systems to backyard gardens, to full-scale farms, aquaponics can do it all.
        - And the best part – You get to harvest both plants and fish from your garden.  Truly raise your entire meal in your backyard instead of using dirt or toxic chemical solutions to grow plants, aquaponics uses highly nutritious fish effluent that contains all the required nutrients for optimum plant growth. Instead of discharging water, aquaponics uses the plants, naturally occurring bacteria, and the media in which they grow in to clean and purify the water, after which it is returned to the fish tank. This water can be reused indefinitely and will only need to be topped-off when it is lost through transpiration from the plants and evaporation.
        """)

        st.image(r"Website Images\Home image 3.png")

    elif choice == "Fish Classifier" and peripheral_active is False:
        st.subheader("Fish Classifier 🐟")
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

        fish_classes = ['Black Sea Sprat',
                        'Gilt-Head Bream',
                        'Horse Mackerel',
                        'Red Mullet',
                        'Red Sea Bream',
                        'Sea Bass',
                        'Shrimp',
                        'Striped Red Mullet',
                        'Trout']

        if image_file is not None:
            file_details = {"filename": image_file.name, "filetype":image_file.type, "filesize":image_file.size}
            st.write(file_details)
            st.image(load_image(image_file))

            content = Image.open(image_file)
            content = tf.keras.preprocessing.image.img_to_array(content)  # pil to cv
            content = load_and_prep_image(content)
            pred_prob = fish_classifier.predict(tf.expand_dims(content, axis=0))
            pred_class = fish_classes[pred_prob.argmax()]

            second = sorted(list(pred_prob[0]))[-2]
            sec_index = (pred_prob == second).argmax()
            sec_pred = fish_classes[sec_index]

            third = sorted(list(pred_prob[0]))[-3]
            third_index = (pred_prob == third).argmax()
            third_pred = fish_classes[third_index]

            st.markdown("<hr>", True)
            st.subheader("Best Prediction")
            st.success(f"Class: {pred_class} | Confidence: {(pred_prob.max() * 100):.2f}%")
            st.write(fish_details(pred_class))

            st.markdown("<hr>", True)
            st.subheader("Second Prediction")
            st.warning(f"Class: {sec_pred} | Confidence: {(second.max() * 100):.2f}%")
            st.write(fish_details(sec_pred))

            st.markdown("<hr>", True)
            st.subheader("Third Prediction")
            st.error(f"Class: {third_pred} | Confidence: {(third.max() * 100):.2f}%")
            st.write(fish_details(third_pred))

    elif choice == "Fish Recommendation (BETA)" and peripheral_active is False:
        st.subheader("Fish Recommendation (BETA)🎣")
        pH = st.slider("pH", 1., 10.)
        temperature = st.slider("Temperature", 1., 35.)
        turbidity = st.slider("Turbidity", 1., 16.)

        fishes = ['katla', 'sing', 'prawn', 'rui', 'koi', 'pangas', 'tilapia',
        'silverCup', 'karpio', 'magur', 'shrimp']

        if st.button("Make Recommendation"):
            test = np.array([pH, temperature, turbidity]).reshape(1, -1)
            st.write("Recomendation: ", fishes[fish_reco.predict(test)[0]])


    elif choice == "Plant Disease Classifier" and peripheral_active is False:
        st.subheader("Plant Disease Classifier 🌱")
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

        disease_classes = ['Apple___Apple_scab',
                           'Apple___Black_rot',
                           'Apple___Cedar_apple_rust',
                           'Apple___healthy',
                           'Blueberry___healthy',
                           'Cherry_(including_sour)___Powdery_mildew',
                           'Cherry_(including_sour)___healthy',
                           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                           'Corn_(maize)___Common_rust_',
                           'Corn_(maize)___Northern_Leaf_Blight',
                           'Corn_(maize)___healthy',
                           'Grape___Black_rot',
                           'Grape___Esca_(Black_Measles)',
                           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                           'Grape___healthy',
                           'Orange___Haunglongbing_(Citrus_greening)',
                           'Peach___Bacterial_spot',
                           'Peach___healthy',
                           'Pepper,_bell___Bacterial_spot',
                           'Pepper,_bell___healthy',
                           'Potato___Early_blight',
                           'Potato___Late_blight',
                           'Potato___healthy',
                           'Raspberry___healthy',
                           'Soybean___healthy',
                           'Squash___Powdery_mildew',
                           'Strawberry___Leaf_scorch',
                           'Strawberry___healthy',
                           'Tomato___Bacterial_spot',
                           'Tomato___Early_blight',
                           'Tomato___Late_blight',
                           'Tomato___Leaf_Mold',
                           'Tomato___Septoria_leaf_spot',
                           'Tomato___Spider_mites Two-spotted_spider_mite',
                           'Tomato___Target_Spot',
                           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                           'Tomato___Tomato_mosaic_virus',
                           'Tomato___healthy']

        if image_file is not None:
            file_details = {"filename": image_file.name, "filetype":image_file.type, "filesize":image_file.size}
            st.write(file_details)
            st.image(load_image(image_file))

            content = Image.open(image_file)
            content = tf.keras.preprocessing.image.img_to_array(content)  # pil to cv
            content = load_and_prep_image(content)
            pred_prob = plant_disease_classifier.predict(tf.expand_dims(content, axis=0))
            pred_class = disease_classes[pred_prob.argmax()]

            second = sorted(list(pred_prob[0]))[-2]
            sec_index = (pred_prob == second).argmax()
            sec_pred = disease_classes[sec_index]

            third = sorted(list(pred_prob[0]))[-3]
            third_index = (pred_prob == third).argmax()
            third_pred = disease_classes[third_index]

            st.write(f"Class: {pred_class} | Confidence: {(pred_prob.max() * 100):.2f}%")
            st.write(f"Class: {sec_pred} | Confidence: {(second.max() * 100):.2f}%")
            st.write(f"Class: {third_pred} | Confidence: {(third.max() * 100):.2f}%")


    elif choice == "About" and peripheral_active is False:
        st.subheader("About")


if __name__ == "__main__":
    main()
