# **Vehicule Detection**

---

**Vehicle Detection Project**

The steps of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Video Implementation

[//]: # (Image References)
[image1_1]: ./output_images/car_no_car_example.png
[image1_2]: ./output_images/car_color_space_hog_features.png
[image2_2_1_1]: ./output_images/bottom_patch.png
[image2_2_1_2]: ./output_images/default_patches.png
[image2_2_1_3]: ./output_images/vehicle_patch.png
[image2_2_1_4]: ./output_images/vehicle_default_patch.png
[image2_3_1]: ./output_images/heatmaps.png
[image2_3_2]: ./output_images/boxed_vehicle.png

All the code is in the Ipython notebook [project](./vehicle_detection.ipynb). 

---

### Histogram of Oriented Gradients (HOG) feature extraction and train a Linear SVM classifier

#### 1. Load images files and split up time series vehicles image images into train and test examples

The code for this step is contained in the 5th code cell of the IPython notebook.

In the vehicle dataset, the GIT folders contain time-serie data. To optimize the classifier we split manually the vehicle images to make sure train and test images are suffient different from one another. I have used the last 20% of the vehicle images as test examples.

![alt text][image1_1]

#### 2. HOG features extraction parameters definition

The HOG extraction parameters are the following:

```python
  colorspace = 'YUV' 
  orient = 11
  pix_per_cell = 16
  cell_per_block = 2
  hog_channel = 'ALL' 
```

This combination of parameters is the result of executions with different combinations of them and a trade-off between accurancy and number of features, 

Table of combinations of HOG parameters:

| Color Channels   | # Orientations  | pixels-per-cells | cells-per-block | # features | Accurancy |
|:----------------:|:---------------:|:----------------:|:---------------:|:----------:|:---------:|
|   RGB ALL        | 9               | 8                | 2               |5295        |94.16%     |
|   HVS ALL        | 9               | 8                | 2               |5295        |96.12%     |
|   HLS ALL        | 9               | 8                | 2               |5295        |95.99%     |
|   YUV ALL        | 9               | 8                | 2               |5295        |96.51%     |
|   YCrCb ALL      | 9               | 8                | 2               |5295        |96.50%     |
|   YUV ALL        | 9               | 16               | 2               |972         |95.92%     |
|   YCrCb ALL      | 9               | 16               | 2               |972         |95.97%     |
|   YUV ALL        | 11              | 8                | 2               |6448        |96.66%     |
|   YCrCb ALL      | 11              | 8                | 2               |6448        |96.76%     |
|   YUV ALL        | 11              | 16               | 2               |1188        |96.07%     |
|   YCrCb ALL      | 11              | 16               | 2               |1188        |96.03%     |



#### 3. HOG features extraction from vehicles and non-vehicle images

The code for this step is contained in third, 4th and 7th code cells of the IPython notebook.

The extraction of HOG features of the vehicle images is done separately for the training examples and the test examples.

I used an invocation to the `hog ()` method of the library `scikit-image`, passing in the value `False` to the parameter `transform_sqrt`, and `True` to the parameter `feature_vector`.


![alt text][image1_2]

#### 4. Split up features images into randomized training and test sets, define labels vectors and shuffle the train features and labels vectors.

The code for this step is contained in 8th and 9th code cells of the IPython notebook.

As the separation of the vehicle images was done by hand, I only had to separate the images of no-vehicle. For this I used the method `train_test_split()` from the `scikit-learn` library.

To shuffle the train features and labels vectors, I used the `shuffle()` method of the `scikit-learn` library.

#### 5. Trainnig and Parameter Tuning the Linear SVM Classifier

The code for this step is contained in 10th to 13th code cells of the IPython notebook.

To tune the Linear SVM vehicle detection model, I used the `GridSearchCV` scikit-learn's parameter tuning algoritm.

I used the next parameter grid:
```python
  param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']}
  ]
```
The optimal parameter combination was `C = 10` and gave an accuracy of `96.49%`.

#### 6. Save the model

The code for this step is contained in 14th code cell of the IPython notebook.

The last step of this phase is save the classifier and the HOG extraction parameters in a `model` and seve it using the mechanism for serializing and de-serializing a Python object structure.

---

### Vehicule Detection

#### 1. Load the model and extract HOG parameters and classifier

The code for this step is contained in 15th code cell of the IPython notebook.

First, I retrive the saved model and extract the HOG parameters and the classifier.

#### 2. Implement a sliding-window technique

The search for vehicles in the image is carried out by means of a sliding-window that runs through the space of the image. This window determines an image that is sent to the classifier that will determine if the image in the window contains a vehicle. Like the images that were used to train the classifier erán of 64x64 pixels, the size of the sliding-window has that same size.

Vehicles have different sizes depending on where they are in relation to the camera car, so we use different scales for the search window. For each scale of the window, you have to scan the image. I have selected 6 scales.
* 1.0
* 1.3
* 1.6
* 2.0
* 2.5
* 2.85

This process is very heavy and we have to try to reduce as much as possible the patches of the image in which you have to look for vehicles.

The process to follow is the following:

##### 1. Create Image Patches

The code for this step is contained in the `get_image_patches()` method in 16th code cell of the IPython notebook.

First, we reduce the search area to the bottom of the image, which is where the vehicles appear. 
Each selected scaled sliding-window restricts the search to a patch of the image of a certain size.

![alt text][image2_2_1_1]

The search windows can be further reduced if we take into account that cars can only appear on the horizon, when the camera car is getting closer to slower cars, or on the sides, when the camera car is being overtaken for other cars. For this reason, we reduce the image patches where to look for vehicles to those three zones, the default patches.

```python
  left = int(64 * 4.65) + 1
  right = w - int(64 * 4.65) + 1 
```

![alt text][image2_2_1_2]

Once a car is detected, we store its position in a car history. In the next frame we will use this position to create a new search patch. We created this new search patch by multiplying the diagonal of the vehicle's position in the previous frame by a factor. In my case this factor is `1.03`. 

![alt text][image2_2_1_3]

Now, if the vehicle detected in the previous frame is inside the lateral zones, and the search has not been activated in the three default zones, the search is extended to the entire area of the margin in which the vehicle is located.

![alt text][image2_2_1_4]

```python
  if len(car_history) > 0:
      pts_previous_car = car_history[-1]
      
  for pts in pts_previous_car:
      pt1 = pts[0]
      pt2 = pts[1]
      # If the car is within the left margin, the search is restricted to the entire left margin
      if pt2[0] < left : # Car is appearing or disappearing on the left
          new_pt1 = [0, 392]
          xpt2 = left
          if not restart: xpt2 += 1
          new_pt2 = [xpt2, 651]
      # If the car is within the right margin, the search is restricted to the entire right margin
      elif pt1[0] > right: # Car is appearing or disappearing on the right
          xpt1 = right
          if not restart: xpt1 -= 1
          new_pt1 = [xpt1, 392]
          new_pt2 = [w, 651]
          
      # If the car is not within the margins, a new search patch is created whose size is calculated 
      # on the diagonal of the car's frame, multiplied by a factor.
      else: 
          dist = int(np.linalg.norm(pts)) + 1
          new_dist = int(dist * 1.03)
          aug = (new_dist - dist) // 2
          new_pt1 = np.array(pts[0]) - aug
          new_pt2 = np.array(pts[1]) + aug
          
      # Execute only if no restart or car outside left-right margin.
      if new_pt2[0] > left and new_pt1[0] < right: 
          if new_pt1[1] < 416 and new_pt2[1] > 480: 
              patches.append((1.0, (392, 480, max(new_pt1[0],0), min(new_pt2[0],w)), 1))
          if new_pt1[1] < 416 and new_pt2[1] > 484: 
              patches.append((1.3, (392, 497, max(new_pt1[0],0), min(new_pt2[0],w)), 1))
          if new_pt1[1] < 420 and new_pt2[1] > 503: 
              patches.append((1.6, (392, 546, max(new_pt1[0],0), min(new_pt2[0],w)), 1))
          if new_pt1[1] < 448 and new_pt2[1] > 528: 
              patches.append((2.0, (392, 590, max(new_pt1[0],0), min(new_pt2[0],w)), 1))
          if new_pt2[1] > 560: 
              patches.append((2.5, (392, 633, max(new_pt1[0],0), min(new_pt2[0],w)), 1))
          if new_pt2[1] > 603: 
              patches.append((2.85, (392, 666, max(new_pt1[0],0), min(new_pt2[0],w)), 1))
```

The search in the default patches is activated once every `15` frames. Or when the cars detected in the previous frame have not been detected in a frame. In the first frame we perform an exhaustive search in the bottom part of the image 
This control is carried out by means of four global variables that are updated in the `pipeline()` method in twenty first code cell of the IPython notebook. 
```python
  first = True
  nframes = 0
  restart = True
  car_history = []            
```

#### 2. Find Cars in an Image Patch

The code for this step is contained in the `find_cars()` method in 17th code cell of the IPython notebook.

For each patch created in the previous step we look for vehicles according to the scale of the patch.

Crop the image of the jframe to the size of the patch
```python

  ystart, ystop, xstart, xstop = shape
 
  img_tosearch = img[ystart:ystop,xstart:xstop,:] 
```
Convert the patch to the specified color scale
```python
  ctrans_tosearch = cv2.cvtColor(img_tosearch, convColor[color_scale])
```
and Scale the image to the specified size
```python
  ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), 
                                                 np.int(imshape[0]/scale)))
```
Calculate the number of blocks and number of steps along x and y, taking into account the size of the window, the parameters of the HOG extraction and the number of cells per step defined for the patch.
```python
  nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
  nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
  
  window = 64
  nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
  nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
  nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1   
```
Compute individual channel HOG features for the entire patch to optimize the number of calculations of the HOG features
```python
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
```
For each possible window in the image patch, extract its HOG features for the window and pass them to the classifier.

If the classifier determines that there is a vehicle in the window, create a box that determines the position of the detected vehicle.
```python
  for xb in range(nxsteps):
      for yb in range(nysteps):
          ypos = yb*cells_per_step
          xpos = xb*cells_per_step
          # Extract HOG for this patch
          hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
          hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
          hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
          hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

          xleft = xpos*pix_per_cell
          ytop = ypos*pix_per_cell

          # Make a prediction
          test_prediction = svc.predict(hog_features.reshape(1,-1))
          
          # Create the box
          if test_prediction == 1 or all_squares:
              xbox_left = np.int(xleft*scale)
              ytop_draw = np.int(ytop*scale)
              win_draw = np.int(window*scale)
              box = [(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)]
              boxes.append(box)
```
---
### Video Implementation
#### 1. Filter false positive

The code for this step is contained in the `add_heat()` and `apply_threshold()`  methods in 18th and 19th code cells of the IPython notebook.

Once we have a list with all the positions of detected vehicles, we create a heatmap adding += 1 for all pixels inside each bbox.
```python
  for box in bbox_list:
      heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1  
```
Then, we eliminate those pixels of the heatmap that have a value below a threshold considering false positives.
```python
  heatmap[heatmap <= threshold] = 0
```

![alt text][image2_3_1]

#### 2. Draw boxes around detected vehicles

The code for this step is contained in the `draw_labeled_bboxes()` and `pipeline()`  methods in 20th and 21st code cells of the IPython notebook.

First, we label the final boxes from heatmap using `label()` function of the library `scipy`.
```python
  labels = label(heatmap)
```

with this labeled map we draw the boxes in the image of the frame.
```python
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        boxes.append(bbox)
```
![alt text][image2_3_2]

Then, the boxes are stored in the car history if they were detected in this frame.
```python
  if len(boxes) > 0 and len(flat_list) > 1: 
      car_history.append(boxes)
```
Here's a [link to project video result](./output_videos/project_video.mp4)

---
### Discussion

A car history has been implemented, with a maximum of three frames, for:
* Obtain the positions of the vehicles detected in the previous frame to reduce the image patch where to look for the vehicles.
* Serve as a sentinel to restart the search in the default patches.
  - When the car history is empty and no vehicle has been detected in the current frame, we restart the search in the default patches.
```python
  if len(car_history) > 0 and len(flat_list) == 0: 
      del car_history[0] 
      if len(car_history) == 0: 
          restart = True
```
  - When fewer cars have been detected than in the previous frame, we restart the search in the default patches.
```python
  if len(car_history) > 0:
      prev_cars = car_history[-1]
      if len(boxes) < len(prev_cars):
          restart = True
```  
* Reinforcement for the detection of vehicles of the current frame. So that a poor detection is not considered false positive.
```python
  if len(car_history) > 0 and len(flat_list) < 2:
      new_list.append(np.concatenate(car_history))
  elif len(car_history) > 0:
      new_list.append(car_history[-1])
```

