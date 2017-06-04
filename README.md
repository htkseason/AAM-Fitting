# AAM-Fitting  
AAM, active appearance model, fitting algrithom. (Include trianing kits)   
This project is based on [VisionModelLib](https://github.com/htkseason/VisionModelLib) and [OpenCV 3.2.0](http://opencv.org/releases.html).  
  
----  
  
__WorkFlow__  
Step 1 -- Train Shape-Model, Texture-Model and Appearance-Model (pre-trained models are included in 'models')  
Step 2 -- Detect face area in the picture, and give an average prediction.  
Step 3 -- Estimate loss and gradient, descent while estimation.  
  
__Demo__  
<img src="https://github.com/htkseason/AAM-Fitting/blob/master/demo.jpg" width="75%" alt="demo" />  
  
----  
  
__References__  
[Cootes and C.J.Taylor. Statistical Models of Appearance for Computer Vision. University of Manchester, March 2004](http://www.face-rec.org/algorithms/AAM/app_models.pdf)  
  
[github.com/MasteringOpenCV](https://github.com/MasteringOpenCV/code)  