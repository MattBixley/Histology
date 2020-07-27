# ml_server

# adapted from
# https://rstudio-conf-2020.github.io/dl-keras-tf/notebooks/project1-natural-images.nb.html

library(keras)
library(tidyverse)
use_condaenv("r-tensorflow")

# define the directories:
# "/Volumes/userdata/staff_groups/merrimanlab/Merriman_Documents/Matt/Histology/"
image_dir <- "data/stomach"
#image_dir <- "/media/xsan/staff_groups/merrimanlab/Merriman_Documents/Matt/Histology/data/stomach"
train_dir <- file.path(image_dir, "train")
test_dir <- file.path(image_dir, "test")

classes <- c("dead", "alive")
total_train <- 330
total_test <- 110
target_size <- c(255,255)
batch <- 8

# simple model for testing
build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

model <- build_model()
model %>% summary()


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
  steps_per_epoch = ceiling(total_train / batch),
  epochs = 75,
  validation_data = test_generator,
  validation_steps = ceiling(total_test / batch),
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
