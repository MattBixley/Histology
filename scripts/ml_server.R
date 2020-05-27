# ml_server

# adapted from
# https://rstudio-conf-2020.github.io/dl-keras-tf/notebooks/project1-natural-images.nb.html

library(keras)
library(tidyverse)

# define the directories:
# "/Volumes/userdata/staff_groups/merrimanlab/Merriman_Documents/Matt/Histology/"
image_dir <- "data/stomach"
train_dir <- file.path(image_dir, "train")
test_dir <- file.path(image_dir, "test")

classes <- c("dead", "alive")
total_train <- 330
total_test <- 110
target_size <- c(255,255)
batch <- 128

for (class in classes) {
  # how many images in each class
  n_train <- length(list.files(file.path(train_dir, class)))
  n_test <- length(list.files(file.path(test_dir, class)))
  
  cat(toupper(class), ": ", 
      "train (", n_train, "), ", 
      "test (", n_test, ")", "\n", sep = "")
  
  # tally up totals
  total_train <- total_train + n_train
  total_test <- total_test + n_test
}

cat("\n", "total training images: ", total_train, "\n",
    "total test images: ", total_test, sep = "")

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", 
                input_shape = c(target_size, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = length(classes), activation = "softmax")

summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "rmsprop",
  metrics = "accuracy"
)

# only augment training data
train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

# do not augment test and validation data
test_datagen <- image_data_generator(rescale = 1/255)

# generate batches of data from training directory
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = target_size,
  batch_size = batch, # edit batch size to smaller than sample size
  class_mode = "categorical"
)

# generate batches of data from validation directory
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = target_size,
  batch_size = batch, # edit batch size to smaller than sample size
  class_mode = "categorical"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = ceiling(total_train / 1),
  epochs = 10,
  validation_data = test_generator,
  validation_steps = ceiling(total_test / 1),
  callbacks = list(
    callback_reduce_lr_on_plateau(patience = 3),
    callback_early_stopping(patience = 7)
  )
)

best_epoch <- which.min(history$metrics$val_loss)
best_loss <- history$metrics$val_loss[best_epoch] %>% round(3)
best_acc <- history$metrics$val_accuracy[best_epoch] %>% round(3)

print(paste0("Our optimal loss is ",best_loss," with an accuracy of ",best_acc))

plot(history) + 
  scale_x_continuous(limits = c(0, length(history$metrics$val_loss)))