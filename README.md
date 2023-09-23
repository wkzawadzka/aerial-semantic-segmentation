# aerial-semantic-segmentation
Binary semantic segmentation approach using U-Net with Keras for indentification of landing safe zones for aerial vehicles.

Final project for Machine Learning in Intelligent Transportation (700.340, 23S) by Assoc. Prof. Pd. Dr. Techn. Habil. Fadi Al Machot at Klagenfurt University.

Binary semantic segmentation is a computer vision problem and refers to the process of visually
separating part of an image into a mask with the value of 1 for the detected presence of a
given class, and 0 for its absence. For a binary semantic segmentation problem that will be
addressed in this report, this process will be used with one class only, pine trees. This specific
project deals with images of landing areas for air vehicles, and the goal is to extract masks for
each image given the presence of pine trees. Landing areas here are forests, which corresponds
to the situation when pilots need to make emergency landings. In such a scenario, accurately
identifying and delineating pine trees in the images is crucial. The main idea for this project is
to follow the model architecture from the original UNet Paper: [U-Net: Convolutional Networks
for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger et al. from 2015 and use it on the
landing areas dataset.

- U-Net network
<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png"  width="620"/>
  
- Examplar input & mask
<img src="https://github.com/wkzawadzka/aerial-semantic-segmentation/assets/49953771/741f5710-09fa-4648-9fd9-92107c32c36b"  width="620"/>
