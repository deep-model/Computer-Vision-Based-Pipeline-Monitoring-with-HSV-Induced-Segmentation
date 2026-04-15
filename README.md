

# Computer Vision Based Pipeline Monitoring with HSV Induced Segmentation

[Veiw Entire Project List](https://github.com/deep-model?tab=repositories)


<img width="800" height="600" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/HSV%20Induced%20Pipeline%20Hot%20Spot%20Detection.png"/>

## Computer Vision Based Applications using HSV Segmentation
HSV-based image segmentation is a powerful and efficient computer vision technique for monitoring asset performance which provides unsupervised and supervised AI rationalization in real-time using thermal forward-looking infrared (FLIR) imagery. Unlike traditional RGB color spaces, HSV (Hue, Saturation, Value) image segmentation separates color information from intensity, making it highly effective for isolating thermal signatures associated with fluid levels, surface temperatures, and structural variations which can be applied to liquid terminal storage tanks and many additional asset monitoring applications. 

## HSV-Induced Image Segmentation Process
The figure below illustrates the HSV image segmentation process for HSV-based image analysis which generally consists of folliwng steps:

1. Capture the video frame or still image to input for processing.
2. Analyze the image and extract hue, saturation, and values for pre-processing.
3. Apply segmentation to the image by analyzing specific objects and areas with HSV thresholding.
4. Identify areas of interest with morphology to contour the segmented object or image. 

<p align="left">
<img width="400" height="300" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/FLIR%20Pipeline%20Image.png"/>
<img width="400" height="300" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/Pipe%20HSV%20Hot%20Mask.png"/>
<img width="400" height="300" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/Pipe%20HSV%20Segmented%20Areas%20of%20Interest.png"/>
<img width="400" height="300" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/HSV%20Induced%20Pipeline%20Hot%20Spot%20Detection.png"/>
</p>


## Color Gradients & Temperature Variations
In thermal imaging, temperature variations are represented through color gradients typically including white and yellow for hot regions, red and orange for warm, and finally purple and blue for cooler areas. By transforming FLIR images into HSV dimensional space, these thermal color signatures can be segmented into distinct regions based on predefined thresholds. For example, warmer fluid regions within a tank may exhibit different thermal characteristics than vapor space or ambient surroundings, enabling indirect level detection through temperature stratification. This is evident in the Hot and Cold HSV saturated images processed in the computer vision model below.

<img width="800" height="600" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/FLIR%20Pipeline%20Image.png"/>


## HSV-Threshold Induced Image Segmentation with Object Masking
In addition, the HSV image segmentation process begins by transforming the incoming thermal frame from RGB to HSV color space. Thresholding is then applied to isolate specific temperature bands by defining upper and lower HSV ranges corresponding to “hot,” “warm,” or “cool” regions. This configuration of color-temperature band thresholding is often pre-configured based on empirical testing and can be fine-tuned for the specific asset for performance enhancement. Once the thresholding is defined, binary masks are produced (e.g., hot masks or cool masks) which box significant areas of interest detected. Further morphological operations such as opening and closing are then applied to remove noise and enhance region continuity and segmentation, ensuring stable detection of tank boundaries and internal thermal zones.

<p align="left">

<img width="400" height="300" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/Pipe%20HSV%20Hot%20Mask.png" />
<img width="400" height="300" alt="image" src="https://github.com/deep-model/Computer-Vision-Based-Pipeline-Monitoring-with-HSV-Induced-Segmentation/blob/main/Pipe%20HSV%20Segmentation%20Mask.png" />

</p>

## Contour Detection & Morphology with Spacial Boundaries
Once segmented, connected-component analysis or contour detection is used to identify tank structures and regions within each tank. By analyzing the spatial distribution of thermal regions—particularly along the vertical axis—autonomous systems can infer level indications. For example, a distinct transition between warm liquid and cooler vapor space can be detected as a boundary within the segmented mask. Additionally, calculating metrics such as the percentage of hot or cool pixels within a defined tank region enables quantitative assessment of thermal distribution and potential anomalies. The image below demonstrates the area of interest with red boxes to highlight hot pixels within the defined tank region for further assessment, in this case tank level reinforcement but can be extended to asset integrity monitoring in many applications.

<img width="800" height="600" alt="image" src="https://github.com/deep-model/HSV-Induced-Image-Segmentation-for-Real-Time-Asset-Monitoring/blob/main/5%20-%20HSV%20Saturatio%20Indeuced%20Hot%20Mask%20with%20Areas%20of%20interest.png"/>

## Applications in Real-Time Asset Monitoring
In real-time applications, HSV image processing can be integrated with live camera feeds to continuously monitor tank asset conditions. The system can dynamically update thresholds, track temporal changes, and trigger alerts when abnormal thermal patterns are detected, such as unexpected variable temperature  changes in cooling (possible emptying or leakage) or overheating (indicating process deviations), leak detection, and dynamic material balancing and process rationalization. This approach is particularly valuable in industrial environments where direct level measurement may be limited or where non-invasive monitoring is preferred and complimentary to inferred measurement as reinforcement. The image below illustrates the computer vision processed image featuring reinforced level measurement based on HSV image segmentation.

<img width="800" height="600" alt="image" src="https://github.com/deep-model/HSV-Induced-Image-Segmentation-for-Real-Time-Asset-Monitoring/blob/main/7%20-%20Tank%20Asset%20Integrity%20Monitoring%20with%20HSV%20Induced%20Segmentation.png"/>

## Conclusion
While HSV-based segmentation does not provide absolute temperature measurements unless radiometric data is available, it offers a robust method for relative thermal analysis and trend detection. When combined with historical baselines and process knowledge, computer vision based HSV analysis enables predictive insights into process and asset behavior, improving operational safety, reliability, and efficiency
