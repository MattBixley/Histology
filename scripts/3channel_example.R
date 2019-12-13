## image calssification example, 3 channels

library(keras)
start <- Sys.time()
fruit_list <- c("Kiwi", "Banana", "Plum", "Apricot", "Avocado", "Cocos", "Clementine", "Mandarine", "Orange",
                "Limes", "Lemon", "Peach", "Plum", "Raspberry", "Strawberry", "Pineapple", "Pomegranate")

# number of output classes (i.e. fruits)
output_n <- length(fruit_list)

# image size to scale down to (original images are 100 x 100 px)
img_width <- 32
img_height <- 32
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
path <- "data/fruits-360"
train_image_files_path <- file.path(path, "Training")
valid_image_files_path <- file.path(path, "Test")

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
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = 'categorical',
                                                    classes = fruit_list,
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = 'categorical',
                                                    classes = fruit_list,
                                                    seed = 42)

cat("Number of images per class:")
table(factor(train_image_array_gen$classes))

cat("\nClass label vs index mapping:\n")
## Class label vs index mapping:
train_image_array_gen$class_indices

### model definition
# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10

# initialise model
model <- keras_model_sequential()

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(img_width, img_height, channels)
)

summary(conv_base)

model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_image_array_gen,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = valid_image_array_gen,
  validation_steps = 50
)

# fit
hist <- fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs,
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint(file.path(path, "fruits_checkpoints.h5"), save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = file.path(path, "fruits_logs"))
  )
)

df_out <- hist$metrics %>% 
{data.frame(acc = .$acc[epochs], val_acc = .$val_acc[epochs], elapsed_time = as.integer(Sys.time()) - as.integer(start))}