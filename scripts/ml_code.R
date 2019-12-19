### Keras transfer learning example uses VGG16 as inception didn't work
# http://flovv.github.io/Logo_detection_transfer_learning/
################### Section 1 #########################
library(keras)
start <- Sys.time()
outcome_list <- c("Censured", "Alive")

# number of output classes 
output_n <- length(outcome_list)

# image size to scale down to (original images are 100 x 100 px)
img_width <- 32
img_height <- 32
target_size <- c(img_width, img_height)
batch_size <- 8

# RGB = 3 channels
channels <- 3

# path to image folders
path <- "data/images"
train_directory <- file.path(path, "Training")
test_directory <- file.path(path, "Test")

# optional data augmentation
train_data_gen = image_data_generator(
  rescale = 1/255 #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fill_mode = "nearest"
)

# Validation data shouldn't be augmented! But it should also be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255
)  

# training images
train_image_array_gen <- flow_images_from_directory(train_directory, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = 'categorical',
                                                    classes = fruit_list,
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(test_directory, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = 'categorical',
                                                    classes = fruit_list,
                                                    seed = 42)


# this next bit needsw to be looked at ofr the actual data, and classes
cat("Number of images per class:")
table(factor(train_image_array_gen$classes))

cat("\nClass label vs index mapping:\n")
## Class label vs index mapping:
train_image_array_gen$class_indices

### model definition
# number of training samples
train_samples <- 10 # train_image_array_gen$n
# number of validation samples
valid_samples <- 10 # valid_image_array_gen$n

################### Section 2 #########################
#base_model <- application_inception_v3(weights = 'imagenet', include_top = FALSE)
base_model <- application_vgg16(weights = 'imagenet', include_top = FALSE)
### use vgg16 -  as inception won't converge --- 

################### Section 3 #########################
## add your custom layers
predictions <- base_model$output %>% 
  layer_global_average_pooling_2d(trainable = T) %>% 
  layer_dense(64, trainable = T) %>%
  layer_activation("relu", trainable = T) %>%
  layer_dropout(0.4, trainable = T) %>%
  layer_dense(27, trainable=T) %>%    ## important to adapt to fit the 27 classes in the dataset!
  layer_activation("softmax", trainable=T)

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)

################### Section 4 #########################
for (layer in base_model$layers)
  layer$trainable <- FALSE

################### Section 5 #########################
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.003, decay = 1e-6),  ## play with the learning rate
  metrics = "accuracy"
)

hist <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = 10, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=2
)

### saveable data frame obejct.
histDF <- data.frame(acc = unlist(hist$history$acc), val_acc=unlist(hist$history$val_acc), val_loss = unlist(hist$history$val_loss),loss = unlist(hist$history$loss))