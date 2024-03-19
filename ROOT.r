library(tidyverse)
library(caret)
library(ROCR)
library(glmnet)
library(rpart)

# ***Helper and Primary Functions***

# Training
train <- function(training_data, outcome, treatment, sample) {
  S <- training_data[[sample]]  # indicator for the sample
  Y <- training_data[[outcome]]  # outcome variable
  T <- training_data[[treatment]]  # indicator for the treatment
  X <- training_data %>%
    select(-c(outcome, treatment, sample))  # pre-treatment covariates

  pi <- mean(S)  # proportion of units in experimental study

  # P(S=1 | X)
  pi_m <- trainControl(method="cv") %>%
    train(
      X,
      (S / pi),
      method="AdaBoost",
      trControl=trainControl(method="cv")
    )

  # P(T=1 | X)
  e_m <- trainControl(method="cv") %>%
    train(
      X[S == 1, ],
      T[S == 1],
      method="LogitBoost",
      trControl=trainControl(method="cv")
    )

  return(list(pi = pi, pi_m = pi_m, e_m = e_m))
}

# Estimation
estimate <- function(testing_data, outcome, treatment, sample, pi, pi_m, e_m) {
  S <- testing_data[[sample]]  # indicator for the sample
  Y <- testing_data[[outcome]]  # outcome variable
  T <- testing_data[[treatment]]  # indicator for the treatment
  X <- testing_data %>%
    select(-c(outcome, treatment, sample))  # pre-treatment covariates

  # pi = P(S=1)
  pi <- mean(S)

  # l(X) = (P(S=1 | X)/P(S=1)) / (P(S=0 | X)/P(S=0))
  lX <- (predict(pi_m, X) / pi) / ((1/pi - predict(pi_m, X)) / (1 - pi))

  # IPW Estimator for ATTE
  a <- ((S * T * Y) / predict(e_m, X, type="response")[, 2]) - 
       ((S * (1 - T) * Y) / predict(e_m, X, type="response")[, 1])
  b <- 1 / lX
  v <- a * b

  return(list(v = v, a = a, b = b))
}

# Estimation for DML
estimate_dml <- function(data, outcome, treatment, sample, crossfit=5) {
  n <- nrow(data)  # total number of units
  df_v <- list()
  skf <- createFolds(data[[sample]], k=crossfit, returnTrain=TRUE)
  
  for (i in seq_along(skf)) {
    cat("Fold", i, "\n")
    
    train_index <- skf[[i]]
    test_index <- setdiff(seq_len(n), train_index)
    
    training_data <- data[train_index, ]
    testing_data <- data[test_index, ]
    
    model <- train(training_data, outcome, treatment, sample)
    v <- estimate(testing_data, outcome, treatment, sample, model$pi, model$pi_m, model$e_m)
    
    df_v_ <- data.frame(te = v$v, primary_index = rownames(testing_data), a = v$a, b = v$b)
    df_v[[i]] <- df_v_
  }
  
  df_v <- bind_rows(df_v)
  df_v <- df_v %>% filter(!is.na(te))
  
  df_v$te_sq <- (df_v$te - mean(df_v$te[data[[sample]] == 1]))^2
  df_v$a_sq <- (df_v$a - mean(df_v$a[data[[sample]] == 1]))^2
  df_v <- df_v %>% group_by(primary_index) %>% summarise_all(mean) %>% filter(data[[sample]] == 1)
  
  data2 <- data[df_v$primary_index, ]
  
  return(list(df_v = df_v, pi = model$pi, pi_m = model$pi_m, e_m = model$e_m, data2 = data2))
}

characterize_tree <- function(X, w, max_depth=3) {
  f <- rpart(w ~ ., data=X, method="class", control=rpart.control(maxdepth=max_depth))
  return(f)
}


midpoint <- function(X) {
  return((max(X) + min(X)) / 2)
}

choose <- function(split_feature, depth) {
  split_prob <- split_feature
  split_prob[1] <- split_prob[1] * (2 ^ (0 * depth / 4))
  split_prob <- split_prob / sum(split_prob)
  fj <- sample(names(split_feature), 1, prob=split_prob)
  return(fj)
}

loss <- function(val, indices, D) {
  D_ <- D
  D_[indices, "w"] <- val
  se <- sqrt(sum(D_$vsq * D_$w) / sum(D_$w)^2)
  se[is.na(se)] <- Inf
  return(se)
}

reduce_weight <- function(fj, split_feature) {
  split_feature[fj] <- split_feature[fj] / 2
  split_feature <- split_feature / sum(split_feature)
  return(split_feature)
}

# Function to split the data into left and right based on the chosen feature and midpoint
split <- function(split_feature, X, D, parent_loss, depth, explore_proba = 0.05) {
  # Choose a feature to split on
  fj <- choose(split_feature, depth)

  # Base case: if fj is a leaf node
  if (fj == "leaf") {
    # Calculate losses for both treatment groups
    losses <- c(loss(0, X$index, D), loss(1, X$index, D))

    # Select exploitation and exploration weights
    w_exploit <- which.min(losses)
    w_explore <- rbinom(1, 1, 0.5)

    # Randomly decide whether to explore
    explore <- rbinom(1, 1, explore_proba)

    # Combine exploitation and exploration weights
    w <- (explore * w_explore) + ((1 - explore) * w_exploit)

    # Update the global and local datasets with the selected weight
    D[X$index, "w"] <- w
    X[X$index, "w"] <- w

    # Return information about the leaf node
    return(list(node = fj, w = w, local_objective = min(losses), depth = depth))
  }
  # Induction case: fj is not a leaf node
  else {
    # Choose the midpoint for the split
    cj <- midpoint(X[[fj]])

    # Split the dataset into left and right based on the chosen feature and midpoint
    X_left <- X[X[[fj]] <= cj, ]
    X_right <- X[X[[fj]] > cj, ]

    # Calculate losses for both treatment groups in the left and right splits
    loss_left <- c(loss(0, X_left$index, D), loss(1, X_left$index, D))
    loss_right <- c(loss(0, X_right$index, D), loss(1, X_right$index, D))

    # Find the minimum losses for left and right splits
    min_loss_left <- min(loss_left)
    min_loss_right <- min(loss_right)

    # Calculate the new loss for the current split
    new_loss <- (nrow(X_left) * min_loss_left + nrow(X_right) * min_loss_right) / nrow(X)

    # Check if the new loss is smaller than the parent loss
    if (new_loss <= parent_loss) {
      # Choose exploitation weights for left and right
      w_left <- which.min(loss_left)
      w_right <- which.min(loss_right)

      # Update the global and local datasets with the selected weights for left and right
      D[X_left$index, "w"] <- w_left
      X_left[X_left$index, "w"] <- w_left

      D[X_right$index, "w"] <- w_right
      X_right[X_right$index, "w"] <- w_right

      # Randomly choose the order of left and right subtrees
      if (rbinom(1, 1, 0.5)) {
        return(list(
          node = fj,
          split = cj,
          left_tree = split(split_feature, X_left, D, new_loss, depth + 1),
          right_tree = split(split_feature, X_right, D, new_loss, depth + 1),
          local_objective = sqrt(sum(D$vsq * D$w) / (sum(D$w)^2)),
          depth = depth
        ))
      } else {
        return(list(
          node = fj,
          split = cj,
          right_tree = split(split_feature, X_right, D, new_loss, depth + 1),
          left_tree = split(split_feature, X_left, D, new_loss, depth + 1),
          local_objective = sqrt(sum(D$vsq * D$w) / (sum(D$w)^2)),
          depth = depth
        ))
      }
    } else {
      # If the new loss is not smaller, update the split feature and try again
      split_feature_updated <- reduce_weight(fj, split_feature)
      return(split(split_feature_updated, X, D, parent_loss, depth))
    }
  }
}

# Function to build the forest of weighted trees
forest_opt <- function(data, outcome, treatment, sample, leaf_proba=0.25, seed=42,
                        num_trees=10, vote_threshold=2 / 3, explore_proba=0.05,
                        feature_est="Ridge", top_k_trees=FALSE, k=10, cutoff="baseline") {
  set.seed(seed)

  # Estimate treatment effects using DML
  df_v <- estimate_dml(data, outcome, treatment, sample, crossfit=5)

  # Extract relevant columns from the testing data
  S <- data[[sample]]  # Indicator for the sample
  Y <- data[[outcome]]  # Outcome variable
  T <- data[[treatment]]  # Indicator for the treatment
  X <- data[, !(names(data) %in% c(outcome, treatment, sample))]  # Pre-treatment covariates
  n <- nrow(data)  # Total number of units
  v <- df_v$te
  vsq <- df_v$te_sq
  cat("ATE Est:", mean(v), "\n")

  # Calculate features for tree splitting
  features <- c("leaf", names(X))

  # Choose the feature importance estimation method
  if (feature_est == "Ridge") {
    vsq_m <- glmnet::cv.glmnet(as.matrix(X), vsq, alpha=1)
    proba <- c(leaf_proba, abs(coef(vsq_m, s="lambda.min")[-1]) / sum(abs(coef(vsq_m, s="lambda.min")[-1])))
  } else {
    vsq_m <- gbm(X, vsq, n.trees=100)
    proba <- c(leaf_proba, vsq_m$var.imp / sum(vsq_m$var.imp))
  }

  proba <- proba / sum(proba)
  split_feature <- data.table(features, proba)

  set.seed(seed)

  w_forest <- list()
  D_forest <- copy(X)
  D_forest[, c("v", "vsq", "S") := .(v, vsq, S)]

  # Add inverse propensity scores to the dataset
  D_forest[, "l(X)" := 1 / df_v$b]

  # Build the forest of weighted trees
  for (t_iter in 1:num_trees) {
    D <- copy(X)
    D[, c("v", "vsq", "w", "S") := .(v, vsq, rep(1, n), S)]
    w_tree <- split(split_feature, D, D, Inf, 0, explore_proba=explore_proba)
    D_forest[, paste("w_tree_", t_iter, sep="") := D$w]
    w_forest[[t_iter]] <- w_tree
  }

  # Select top k trees if specified
  if (top_k_trees) {
    obj_trees <- sapply(w_forest, function(x) x$local_objective)
    idx <- order(obj_trees)[1:k]
    rashomon_set <- idx
    not_in_rashomon_set <- setdiff(seq_len(num_trees), idx)
  } else {
    # Select trees based on the cutoff value
    if (cutoff == "baseline") {
      baseline_loss <- sqrt(sum(D_forest$vsq) / (n^2))
      cutoff <- baseline_loss
    }
    not_in_rashomon_set <- seq_len(num_trees)[w_forest[[i]]$local_objective >= cutoff]
    rashomon_set <- seq_len(num_trees)[w_forest[[i]]$local_objective < cutoff]
  }

  # Create datasets with and without the selected trees
  D_rash <- D_forest[, !(paste0("w_tree_", not_in_rashomon_set) %in% names(D_forest)), with=FALSE]
  D_w_rash <- D_forest[, ..paste0("w_tree_", rashomon_set)]
  D_rash[, w_opt := as.integer(rowMeans(D_w_rash) > vote_threshold)]
  D_rash[, vote_count := rowSums(D_w_rash)]

  # Characterize the final tree ensemble
  f <- characterize_tree(X, D_rash$w_opt)

  return(list(D_rash, D_forest, w_forest, rashomon_set, f, data))
}





