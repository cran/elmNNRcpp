## ---- eval=T-------------------------------------------------------------

# load the data and split it in two parts
#----------------------------------------

data(Boston, package = 'KernelKnn')

library(elmNNRcpp)

Boston = as.matrix(Boston)
dimnames(Boston) = NULL

X = Boston[, -dim(Boston)[2]]
xtr = X[1:350, ]
xte = X[351:nrow(X), ]


# prepare / convert the train-data-response to a one-column matrix
#-----------------------------------------------------------------

ytr = matrix(Boston[1:350, dim(Boston)[2]], nrow = length(Boston[1:350, dim(Boston)[2]]),
             
             ncol = 1)


# perform a fit and predict [ elmNNRcpp ]
#----------------------------------------

fit_elm = elm_train(xtr, ytr, nhid = 1000, actfun = 'purelin',
                    
                    init_weights = "uniform_negative", bias = TRUE, verbose = T)

pr_te_elm = elm_predict(fit_elm, xte)



# perform a fit and predict [ lm ]
#----------------------------------------

data(Boston, package = 'KernelKnn')

fit_lm = lm(medv~., data = Boston[1:350, ])

pr_te_lm = predict(fit_lm, newdata = Boston[351:nrow(X), ])



# evaluation metric
#------------------

rmse = function (y_true, y_pred) {
  
  out = sqrt(mean((y_true - y_pred)^2))
  
  out
}


# test data response variable
#----------------------------

yte = Boston[351:nrow(X), dim(Boston)[2]]


# mean-squared-error for 'elm' and 'lm'
#--------------------------------------

cat('the rmse error for extreme-learning-machine is :', rmse(yte, pr_te_elm[, 1]), '\n')

cat('the rmse error for liner-model is :', rmse(yte, pr_te_lm), '\n')


## ---- eval=T-------------------------------------------------------------


# load the data
#--------------

data(ionosphere, package = 'KernelKnn')

y_class = ionosphere[, ncol(ionosphere)]

x_class = ionosphere[, -c(2, ncol(ionosphere))]     # second column has 1 unique value

x_class = scale(x_class[, -ncol(x_class)])

x_class = as.matrix(x_class)                        # convert to matrix
dimnames(x_class) = NULL 



# split data in train-test
#-------------------------

xtr_class = x_class[1:200, ]                    
xte_class = x_class[201:nrow(ionosphere), ]

ytr_class = as.numeric(y_class[1:200])
yte_class = as.numeric(y_class[201:nrow(ionosphere)])

ytr_class = onehot_encode(ytr_class - 1)                                     # class labels should begin from 0 (subtract 1)


# perform a fit and predict [ elmNNRcpp ]
#----------------------------------------

fit_elm_class = elm_train(xtr_class, ytr_class, nhid = 1000, actfun = 'relu',
                          
                          init_weights = "uniform_negative", bias = TRUE, verbose = TRUE)

pr_elm_class = elm_predict(fit_elm_class, xte_class, normalize = FALSE)

pr_elm_class = max.col(pr_elm_class, ties.method = "random")



# perform a fit and predict [ glm ]
#----------------------------------------

data(ionosphere, package = 'KernelKnn')

fit_glm = glm(class~., data = ionosphere[1:200, -2], family = binomial(link = 'logit'))

pr_glm = predict(fit_glm, newdata = ionosphere[201:nrow(ionosphere), -2], type = 'response')

pr_glm = as.vector(ifelse(pr_glm < 0.5, 1, 2))


# accuracy for 'elm' and 'glm'
#-----------------------------

cat('the accuracy for extreme-learning-machine is :', mean(yte_class == pr_elm_class), '\n')

cat('the accuracy for glm is :', mean(yte_class == pr_glm), '\n')


## ---- eval = F, echo = T-------------------------------------------------
#  
#  
#  # using system('wget..') on a linux OS
#  #-------------------------------------
#  
#  system("wget https://raw.githubusercontent.com/mlampros/DataSets/master/mnist.zip")
#  
#  mnist <- read.table(unz("mnist.zip", "mnist.csv"), nrows = 70000, header = T,
#  
#                      quote = "\"", sep = ",")
#  
#  x = mnist[, -ncol(mnist)]
#  
#  y = mnist[, ncol(mnist)]
#  
#  y_expand = onehot_encode(y)
#  
#  
#  
#  # split the data randomly in train-test
#  #--------------------------------------
#  
#  idx_train = sample(1:nrow(y_expand), round(0.85 * nrow(y_expand)))
#  
#  idx_test = setdiff(1:nrow(y_expand), idx_train)
#  
#  fit = elm_train(as.matrix(x[idx_train, ]), y_expand[idx_train, ], nhid = 2500,
#  
#                  actfun = 'relu', init_weights = 'uniform_negative', bias = TRUE,
#  
#                  verbose = TRUE)
#  
#  
#  # Input weights will be initialized ...
#  # Dot product of input weights and data starts ...
#  # Bias will be added to the dot product ...
#  # 'relu' activation function will be utilized ...
#  # The computation of the Moore-Pseudo-inverse starts ...
#  # The computation is finished!
#  #
#  # Time to complete : 1.607153 mins
#  
#  
#  # predictions for test-data
#  #--------------------------
#  
#  pr_test = elm_predict(fit, newdata = as.matrix(x[idx_test, ]))
#  
#  pr_max_col = max.col(pr_test, ties.method = "random")
#  
#  y_true = max.col(y_expand[idx_test, ])
#  
#  
#  cat('Accuracy ( Mnist data ) :', mean(pr_max_col == y_true), '\n')
#  
#  # Accuracy ( Mnist data ) : 96.13
#  

