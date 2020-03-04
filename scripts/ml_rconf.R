# notes from rconf ML workshop
# https://rstudio-conf-2020.github.io/dl-keras-tf/notebooks/project1-natural-images.nb.html

library(keras)
library(ggplot2)
library(glue)

# define the directories:
image_dir <- here::here("data", "stomach")
train_dir <- file.path(image_dir, "training")
valid_dir <- file.path(image_dir, "validation")
test_dir <- file.path(image_dir, "test")

classes <- c("censured", "alive")
total_train <- 2
total_valid <- 2
total_test <- 2

for (class in classes) {
  # how many images in each class
  n_train <- length(list.files(file.path(train_dir, class)))
  n_valid <- length(list.files(file.path(valid_dir, class)))
  n_test <- length(list.files(file.path(test_dir, class)))
  
  cat(toupper(class), ": ", 
      "train (", n_train, "), ", 
      "valid (", n_valid, "), ", 
      "test (", n_test, ")", "\n", sep = "")
  
  # tally up totals
  total_train <- total_train + n_train
  total_valid <- total_valid + n_valid
  total_test <- total_test + n_test
}

cat("\n", "total training images: ", total_train, "\n",
    "total validation images: ", total_valid, "\n",
    "total test images: ", total_test, sep = "")

op <- par(mfrow = c(2, 4), mar = c(0.5, 0.2, 1, 0.2))
for (class in classes) {
  image_path <- list.files(file.path(train_dir, class), full.names = TRUE)[[1]]
  plot(as.raster(tiff::readTIFF(image_path, native = TRUE)))
  title(main = class)
}

par(op)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", 
                input_shape = c(150, 150, 3)) %>%
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
  target_size = c(150, 150),
  batch_size = 8, # edit batch size to smaller than sample size
  class_mode = "categorical"
)

# generate batches of data from validation directory
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 8, # edit batch size to smaller than sample size
  class_mode = "categorical"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = ceiling(total_train / 1),
  epochs = 50,
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

glue("Our optimal loss is {best_loss} with an accuracy of {best_acc}")

plot(history) + 
  scale_x_continuous(limits = c(0, length(history$metrics$val_loss)))


#### vgg
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

summary(conv_base)

datagen <- image_data_generator(rescale = 1/255)
batch_size <- 1

extract_features <- function(directory, sample_count, shuffle = TRUE) {
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count, length(classes)))
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = shuffle
  )
  i <- 0
  while (TRUE) {
    cat("Processing batch", i + 1, "of", ceiling(sample_count / batch_size), "\n")
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range, ] <- labels_batch
    i <- i + 1
    if (i * batch_size >= sample_count) break
  }
  list(
    features = features,
    labels = labels
  ) 
}

train <- extract_features(train_dir, 32*129)
#validation <- extract_features(valid_dir, 32*43)
test <- extract_features(test_dir, 32*43, shuffle = FALSE)

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
#validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(train$features)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 8, activation = "softmax")

summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001),
  metrics = "accuracy"
)

history_pretrained <- model %>% fit(
  train$features, train$labels,
  epochs = 5,
  batch_size = 1,
  #validation_data = list(validation$features, validation$labels),
  callbacks = list(
    callback_reduce_lr_on_plateau(patience = 3),
    callback_early_stopping(patience = 7)
  )
)

best_epoch <- which.min(history_pretrained$metrics$val_loss)
best_loss <- history_pretrained$metrics$val_loss[best_epoch] %>% round(3)
best_acc <- history_pretrained$metrics$val_accuracy[best_epoch] %>% round(3)

glue("Our optimal loss is {best_loss} with an accuracy of {best_acc}")

plot(history_pretrained) + 
  scale_x_continuous(limits = c(0, length(history_pretrained$metrics$val_loss)))


