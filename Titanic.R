# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# choose a work directory
mywd = "C:/ ... /Downloads"
# mywd = "C:/ ... /Downloads"
setwd(mywd)

# create a name for a .txt file to log progress information while parallel processing
myfile = "log.txt"
file.create(myfile)

# cross validation folds
K = 2

# cross validation replications per fold
R = 5

# -----------------------------------------------------------------------------------
# ---- Packages ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# these are the packages i use

# data handling
require(data.table)
require(stringr)
require(tm)
require(stringdist)
require(gtools)
require(psych)

# plotting
require(VIM)
require(ggplot2)
require(gridExtra)
require(scales)
require(corrplot)
require(factoextra)

# modeling
require(forecast)
require(ranger)
require(e1071)
require(glmnet)
require(pROC)
require(caret)
require(cvTools)
require(SuperLearner)
require(xgboost)
require(h2o)
require(MLmetrics)

# parallel computing
require(foreach)
require(parallel)
require(doSNOW)
require(rlecuyer)

}

# -----------------------------------------------------------------------------------
# ---- Functions --------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- prints the data types of each column in a data frame -------------------------

types = function(dat)
{
  dat = data.frame(dat)
  
  column = sapply(1:ncol(dat), function(i) colnames(dat)[i])
  data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
  levels = sapply(1:ncol(dat), function(i) length(levels(dat[,i])))
  
  return(data.frame(column, data.type, levels))
}

# ---- a qualitative color scheme ---------------------------------------------------

mycolors = function(n)
{
  require(grDevices)
  return(colorRampPalette(c("#e41a1c", "#0099ff", "#4daf4a", "#984ea3", "#ff7f00", "#ff96ca", "#a65628"))(n))
}

# ---- generates a logarithmically spaced sequence ----------------------------------

lseq = function(from, to, length.out)
{
  return(exp(seq(log(from), log(to), length.out = length.out)))
}

# ---- builds a square confusion matrix ---------------------------------------------

confusion = function(ytrue, ypred)
{
  require(gtools)
  
  # make predicted and actual vectors into factors, if they aren't already
  if(class(ytrue) != "factor") ytrue = factor(ytrue)
  if(class(ypred) != "factor") ypred = factor(ypred)
  
  # combine their levels into one unique set of levels
  common.levels = mixedsort(unique(c(levels(ytrue), levels(ypred))))
  
  # give each vector the same levels
  ytrue = factor(ytrue, levels = common.levels)
  ypred = factor(ypred, levels = common.levels)
  
  # return a square confusion matrix
  return(table("Actual" = ytrue, "Predicted" = ypred))
}

# ---- runs goodness of fit tests across all columns of two data sets ---------------

sample.test = function(dat.sample, dat.remain, alpha = 0.5)
{
  # set up the types() function
  # this function extracts the column names, data types, and number of factor levels for each column of a data set
  types = function(dat)
  {
    dat = data.frame(dat)
    
    column = sapply(1:ncol(dat), function(i) colnames(dat)[i])
    data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
    levels = sapply(1:ncol(dat), function(i) length(levels(dat[,i])))
    
    return(data.frame(column, data.type, levels))
  }
  
  # make the data sets into data frames
  dat.sample = data.frame(dat.sample)
  dat.remain = data.frame(dat.remain)
  
  # get the data types of the data sets
  sample.types = types(dat.sample)
  remain.types = types(dat.remain)
  
  # ensure these data sets are identical
  if(identical(sample.types, remain.types))
  {
    # extract the column postion of factor variables
    factor.id = which(sample.types$data.type == "factor")
    
    # extract the column postion of numeric variables
    numeric.id = which(sample.types$data.type == "numeric" | sample.types$data.type == "integer")
    
    # get the p-values for the factor variables
    factor.test = lapply(factor.id, function(i)
    {
      # get the probability of each level of a factor occuring in dat.remain
      prob = as.numeric(table(dat.remain[,i]) / length(dat.remain[,i]))
      
      # get the frequency of each level of a factor occuring in dat.sample
      tab = table(dat.sample[,i])
      
      # perform a chi.sq test to reject or fail to reject the null hypothesis
      # the null: the observed frequency (tab) is equal to the expected count (prob)
      p.val = chisq.test(tab, p = prob)$p.value
      
      # determine if these variables are expected to come from the same distribution
      same.distribution = p.val > alpha
      
      # build a summary for variable i
      output = data.frame(variable = colnames(dat.sample)[i],
                          class = "factor",
                          gof.test = "chisq.test",
                          p.value = p.val,
                          alpha = alpha,
                          same.distribution = same.distribution)
      
      return(output)
    })
    
    # merge the list of rows into one table
    factor.test = do.call("rbind", factor.test)
    
    # get the p-values for the numeric variables
    numeric.test = lapply(numeric.id, function(i)
    {
      # perform a ks test to reject or fail to reject the null hypothesis
      # the null: the two variables come from the same distribution
      p.val = ks.test(dat.sample[,i], dat.remain[,i])$p.value
      
      # determine if these variables are expected to come from the same distribution
      same.distribution = p.val > alpha
      
      # build a summary for variable i
      output = data.frame(variable = colnames(dat.sample)[i],
                          class = "numeric",
                          gof.test = "ks.test",
                          p.value = p.val,
                          alpha = alpha,
                          same.distribution = same.distribution)
      
      return(output)
    })
    
    # merge the list of rows into one table
    numeric.test = do.call("rbind", numeric.test)
    
    # combine the test results into one table
    output = rbind(factor.test, numeric.test)
    
    return(output)
    
  } else
  {
    print("dat.sample and dat.remain must have the same:\n
          1. column names\n
          2. data class for each column\n
          3. number of levels for each factor column")
  }
  }

# ---- creates an array for spliting up rows of a data set for cross validation -----

cv.folds = function(n, K, R, seed)
{
  # load required packages
  require(cvTools)
  require(data.table)
  
  # set the seed for repeatability
  set.seed(seed)
  
  # create the folds for repeated cross validation
  cv = cvFolds(n = n, K = K, R = R)
  
  # extract the fold id (which) and replication id (subsets)
  cv = data.table(cbind(cv$which, cv$subsets))
  
  # rename columns accordingly
  cv.names = c("fold", paste0("rep", seq(1:R)))
  setnames(cv, cv.names)
  
  # create the combinations of folds and replications
  # this is to make sure each fold is a test set once, per replication
  comb = expand.grid(fold = 1:K, rep = 1:R)
  
  # create a list, where each element is also a list where an element indicates which observations are in the training set and testing set for a model
  cv = lapply(1:nrow(comb), function(i)
  {
    # create the testing set
    testing = cv[fold == comb$fold[i]][[comb$rep[i] + 1]]
    
    # create the training set
    training = cv[fold != comb$fold[i]][[comb$rep[i] + 1]]
    
    # return the results in a list
    return(list(train = training, test = testing))
  })
  
  return(cv)
}

# ---- fast missing value imputation by chained random forests ----------------------

# got this from:
# https://github.com/mayer79/missRanger/blob/master/R/missRanger.R

missRanger <- function(data, maxiter = 10L, pmm.k = 0, seed = NULL, ...)
{
  cat("Missing value imputation by chained random forests")
  
  data = data.frame(data)
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  allVars <- names(which(sapply(data, function(z) (is.factor(z) || is.numeric(z)) && any(!is.na(z)))))
  
  if (length(allVars) < ncol(data)) {
    cat("\n  Variables ignored in imputation (wrong data type or all values missing: ")
    cat(setdiff(names(data), allVars), sep = ", ")
  }
  
  stopifnot(length(allVars) > 1L)
  data.na <- is.na(data[, allVars, drop = FALSE])
  count.seq <- sort(colMeans(data.na))
  visit.seq <- names(count.seq)[count.seq > 0]
  
  if (!length(visit.seq)) {
    return(data)
  }
  
  k <- 1L
  predError <- rep(1, length(visit.seq))
  names(predError) <- visit.seq
  crit <- TRUE
  completed <- setdiff(allVars, visit.seq)
  
  while (crit && k <= maxiter) {
    cat("\n  missRanger iteration ", k, ":", sep = "")
    data.last <- data
    predErrorLast <- predError
    
    for (v in visit.seq) {
      v.na <- data.na[, v]
      
      if (length(completed) == 0L) {
        data[, v] <- imputeUnivariate(data[, v])
      } else {
        fit <- ranger(formula = reformulate(completed, response = v), 
                      data = data[!v.na, union(v, completed)],
                      ...)
        pred <- predict(fit, data[v.na, allVars])$predictions
        data[v.na, v] <- if (pmm.k) pmm(fit$predictions, pred, data[!v.na, v], pmm.k) else pred
        predError[[v]] <- fit$prediction.error / (if (fit$treetype == "Regression") var(data[!v.na, v]) else 1)
        
        if (is.nan(predError[[v]])) {
          predError[[v]] <- 0
        }
      }
      
      completed <- union(completed, v)
      cat(".")
    }
    
    cat("done")
    k <- k + 1L
    crit <- mean(predError) < mean(predErrorLast)
  }
  
  cat("\n")
  if (k == 2L || (k == maxiter && crit)) data else data.last
}

}

# -----------------------------------------------------------------------------------
# ---- Prepare Data -----------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Checking Data Types ----------------------------------------------------------

{

# import the data
# for column descriptions see: https://www.kaggle.com/c/titanic/data
train = data.table(read.csv("train.csv", na.strings = ""))
test = data.table(read.csv("test.csv", na.strings = ""))

# lets check out train
train
types(train)

# update columns that should be treated as a different data type
train[, Survived := factor(Survived)]
train[, Pclass := factor(Pclass)]

# lets check out test
test
types(test)

# update columns that should be treated as factors, not numbers
test[, Pclass := factor(Pclass)]

# lets compute the number of cases (ie. survivals) and the population size
cases = sum(as.numeric(as.character(train$Survived)))
population = nrow(train)

# here's the expected proportion of survivors
cases / population

}

# ---- Check for Missing Values -----------------------------------------------------

{

# lets check out if there are any missing values (NA's) in train
aggr(train, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)

# Cabin has so many missing values that this may just mean passengers didn't have a cabin, lets fill in blanks with U (stands for Unknown)
train[is.na(Cabin), Cabin := "U"]

# lets check out if there are any missing values (NA's) in test
aggr(test, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)

# Cabin has so many missing values that this may just mean passengers didn't have a cabin, lets fill in blanks with U (stands for Unknown)
test[is.na(Cabin), Cabin := "U"]

# between the training and testing data sets we have missing values to impute for Age, Embarked, and Fare
missing.vars = c("Age", "Embarked", "Fare")

# lets combine train and test for imputations
dat = data.table(rbind(train[, !"Survived"], test))

}

# ---- Imputations ------------------------------------------------------------------

{

# create a copy of dat
dat.impute = data.table(dat)

# remove Name, Ticket, and Cabin as variables becuase they have too many factor levels (ie. too many degrees of freedom)
# remove PassengerId becuase it is just an ID column, not a predictor
dat.impute[, c("PassengerId", "Name", "Ticket", "Cabin") := NULL]

# use a random forest to impute the missing values
set.seed(42)
rf.impute = missForest(dat.impute, ntree = 500, nodesize = c(5, 1))

# extract the imputed data set
dat.impute = rf.impute$ximp

# only keep variables in missing.vars for dat.impute
dat.impute = dat.impute[, missing.vars, with = FALSE]

# give dat.impute its PassengerId column
dat.impute[, PassengerId := 1:nrow(dat.impute)]

# remove variables in missing.vars from train and test
train = train[, !missing.vars, with = FALSE]
test = test[, !missing.vars, with = FALSE]

# make PassengerId the key column in dat.impute, train, and test for joining purposes
setkey(dat.impute, PassengerId)
setkey(train, PassengerId)
setkey(test, PassengerId)

# join dat.impute onto train and onto test
train = dat.impute[train]
test = dat.impute[test]

# lets remove objects we no longer need
rm(dat.impute, dat, missing.vars, rf.impute)

# free memory
gc()

}

}

# -----------------------------------------------------------------------------------
# ---- Feature Engineering ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# lets combine train and test to do feature engineering on all variable values
dat = data.table(rbind(train[,!"Survived"], test))

# ---- Cabin  -----------------------------------------------------------------------

{

# lets aggregate Cabin by its letter (ie. drop the number) to reduce its levels
dat[, Cabin := factor(gsub("[0-9]", "", Cabin))]

# check out the levels of cabin
levels(dat$Cabin)

# the level T in dat looks weird, lets see how many times it appeared
# its weird because it breaks the cabin lettering pattern by jumping from G to T
nrow(dat[Cabin == "T"])

# it only appears once, so perhaps its a mistake
# lets set it to U
dat[Cabin == "T", Cabin := "U"]
dat[, Cabin := droplevels(Cabin)]

# we can see that some passengers have multiple cabins, so lets make that a variable called Rooms
levels(dat$Cabin)

# remove all spaces
dat[, Cabin := factor(gsub(" ", "", Cabin))]

# count the number of characters to indicate room count
dat[, Rooms := nchar(as.character(Cabin))]

# lets also keep just one Cabin letter for each passenger
# we'll just use the first cabin assignment
dat[, Cabin := factor(substring(Cabin, 1, 1))]

# verify that the levels of Cabin are just single letters
levels(dat$Cabin)

}

# ---- Name -------------------------------------------------------------------------

{

# ---- frequent words ---------------------------------------------------------------

{

# lets text mine the names column to find common words
# we want to find common words that can aggregate names into simplier column(s)

# make all letters uppercase
Name = toupper(as.character(dat$Name))

# remove all punctuation
Name = removePunctuation(Name)

# remove all numbers
Name = removeNumbers(Name)

# merge all words into one line
Name.decomp = paste(Name, collapse = " ")

# split words by a space to create a vector of every word
Name.decomp = strsplit(Name.decomp, " ")[[1]]

# count the frequency of words
Name.table = sort(table(Name.decomp), decreasing = TRUE)

# make Name.table into a table
Name.table = data.table(Name.table)
setnames(Name.table, c("word", "count"))

# make the column word into a factor for plotting purposes
Name.table[, word := factor(word, levels = unique(word))]

# plot the proportion of presence for the top N words
N = 3

ggplot(Name.table[1:N], aes(x = word, y = count / nrow(dat))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = percent) +
  labs(x = "Word", y = "Proportion of Presence") +
  theme_bw(15) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5))

# for each of the top N words create a binary variable for their presence
name.vars = foreach(i = 1:N, .combine = "cbind") %do%
{
  # get the word
  var = Name.table$word[i]
  
  # determine where it is present
  presence = as.numeric(grepl(paste0("\\<", var, "\\>"), Name))
  
  return(presence)
}

# make name.vars into a data.table and give it proper column names
name.vars = data.table(name.vars)
setnames(name.vars, gsub(" ", ".", Name.table$word[1:N]))

# combine name.vars onto dat
dat = cbind(dat, name.vars)

}

# ---- similar names ----------------------------------------------------------------

{

# remove all spacing
Name = gsub(" ", "", Name)

# keep the first 4 letters of each name to simplify clustering
min(nchar(Name))
Name = substr(Name, 1, 4)

# create a string distance matrix
dmat = as.dist(stringdistmatrix(a = Name, b = Name))

# plot dmat
# fviz_dist(dmat, gradient = list(low = "#b31b1b", mid = "white", high = "cornflowerblue")) + 
  ggtitle("\nName Distance Matrix") + 
  theme_void(20) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())

# determine which hierarchical clustering method best summarizes dmat
# we will determine this by comparing the correlation between the original distances and the cophenetic distances

# choice of hclust methods
methods = c("ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid")

# build cluster methods
hc = lapply(methods, function(i) hclust(dmat, method = i))

# compute correlations
cors = rbindlist(lapply(1:length(methods), function(i) 
  data.table(method = methods[i], value = cor(dmat, cophenetic(hc[[i]])))))

# order by value
cors = cors[order(value, decreasing = TRUE)]

# make method a factor for plotting purposes
cors[, method := factor(method, levels = unique(method))]

# plot correlations
ggplot(cors, aes(x = method, y = value, fill = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "method", y = "correlation") +
  scale_fill_gradient(low = "yellow", high = "red") +
  theme_dark(15) +
  theme(legend.position = "none")

# lets go with average
hc = hclust(dmat, method = "average")

# plot the cophenetic distance matrix
# fviz_dist(cophenetic(hc), gradient = list(low = "#b31b1b", mid = "white", high = "cornflowerblue")) + 
  ggtitle("\nCophenetic Distance Matrix") + 
  theme_void(20) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())

# lets try up to k cluster sizes
k = 2:10

# lets plot barplots of how many data points are in each cluster
par(mfrow = c(3, 3))
lapply(k, function(i) barplot(table(cutree(hc, k = i)), main = paste(i, "Clusters"), ylim = c(0, length(Name))))

# plot the dendrogram
par(mfrow = c(1, 1))
plot(hc, hang = -1, labels = FALSE)

# lets go with k clusters
k = 6
table(cutree(hc, k = k))

# plot the dendrogram with k clusters
par(mfrow = c(1, 1))
plot(hc, hang = -1, labels = FALSE)
rect.hclust(hc, k = k, border = "red")

# extract the cluster grouping
Ncode = cutree(hc, k = k)

# give Ncode to dat
dat[, Ncode := factor(Ncode, levels = sort(unique(Ncode)))]

# remove objects we no longer need
rm(cors, dmat, hc, k, methods, Ncode, Name, i, N, Name.table, name.vars, Name.decomp, presence, var)

# free memory
gc()

# remove Name from dat
dat[, Name := NULL]

}

}

# ---- Ticket -----------------------------------------------------------------------

{

# ---- similar coding ---------------------------------------------------------------

{

# make all letters uppercase
Ticket = toupper(as.character(dat$Ticket))

# remove all punctuation
Ticket = removePunctuation(Ticket)

# remove all spacing
Ticket = gsub(" ", "", Ticket)

# keep the first 3 characters of each ticket to simplify clustering
min(nchar(Ticket))
Ticket = substr(Ticket, 1, 3)

# create a string distance matrix
dmat = as.dist(stringdistmatrix(a = Ticket, b = Ticket))

# plot dmat
# fviz_dist(dmat, gradient = list(low = "#b31b1b", mid = "white", high = "cornflowerblue")) + 
  ggtitle("\nTicket Distance Matrix") + 
  theme_void(20) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())

# determine which hierarchical clustering method best summarizes dmat
# we will determine this by comparing the correlation between the original distances and the cophenetic distances

# choice of hclust methods
methods = c("ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid")

# build cluster methods
hc = lapply(methods, function(i) hclust(dmat, method = i))

# compute correlations
cors = rbindlist(lapply(1:length(methods), function(i) 
  data.table(method = methods[i], value = cor(dmat, cophenetic(hc[[i]])))))

# order by value
cors = cors[order(value, decreasing = TRUE)]

# make method a factor for plotting purposes
cors[, method := factor(method, levels = unique(method))]

# plot correlations
ggplot(cors, aes(x = method, y = value, fill = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "method", y = "correlation") +
  scale_fill_gradient(low = "yellow", high = "red") +
  theme_dark(15) +
  theme(legend.position = "none")

# lets go with average
hc = hclust(dmat, method = "average")

# plot the cophenetic distance matrix
# fviz_dist(cophenetic(hc), gradient = list(low = "#b31b1b", mid = "white", high = "cornflowerblue")) + 
  ggtitle("\nCophenetic Distance Matrix") + 
  theme_void(20) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())

# lets try up to k cluster sizes
k = 2:10

# lets plot barplots of how many data points are in each cluster
par(mfrow = c(3, 3))
lapply(k, function(i) barplot(table(cutree(hc, k = i)), main = paste(i, "Clusters"), ylim = c(0, length(Ticket))))

# plot the dendrogram
par(mfrow = c(1, 1))
plot(hc, hang = -1, labels = FALSE)

# lets go with k clusters
k = 8
table(cutree(hc, k = k))

# plot the dendrogram with k clusters
par(mfrow = c(1, 1))
plot(hc, hang = -1, labels = FALSE)
rect.hclust(hc, k = k, border = "red")

# extract the cluster grouping
Tcode = cutree(hc, k = k)

# give Tcode to dat
dat[, Tcode := factor(Tcode, levels = sort(unique(Tcode)))]

# remove objects we no longer need
rm(cors, dmat, hc, k, methods, Tcode, Ticket)

# free memory
gc()

}

# ---- similar size -----------------------------------------------------------------

{

# extract the letters from tickets in dat
ticket.letters = str_extract(dat$Ticket, "[A-z]+")

# extract the numbers from tickets in dat
ticket.numbers = str_extract(dat$Ticket, "[[:digit:]]+")

# count the number of letters and numbers
ticket.letters = nchar(gsub(" ", "", ticket.letters))
ticket.numbers = nchar(gsub(" ", "", ticket.numbers))

# lets add two columns indicating if a ticket does or doesn't have letters/numbers
has.letters = as.numeric(!is.na(ticket.letters))
has.numbers = as.numeric(!is.na(ticket.numbers))

# replace NA's with 0's
ticket.letters[is.na(ticket.letters)] = 0
ticket.numbers[is.na(ticket.numbers)] = 0

# compute total ticket size
ticket.size = ticket.letters + ticket.numbers

# give has.letters, has.numbers, and ticket.size to dat
dat[, Tletters := has.letters]
dat[, Tnumbers := has.numbers]
dat[, Tsize := ticket.size]

# remove Ticket as a column in dat
dat[, Ticket := NULL]

# remove objects we no longer need
rm(ticket.letters, ticket.numbers, has.letters, has.numbers, ticket.size)

# free memory
gc()

}

}

# ---- Family -----------------------------------------------------------------------

{

# lets create the family size of each passenger using SibSp and Parch
dat[, Fsize := SibSp + Parch + 1]

}

# ---- Passengers -------------------------------------------------------------------

{

# lets try to cluster each passenger based on their Age, Fare
dat.cluster = data.table(dat[,.(Age, Fare)])

# put all variables on the same scale for fair comparison
dat.cluster = data.table(scale(dat.cluster))

# build the distance matrix of dat.cluster
dat.dis = get_dist(dat.cluster, method = "euclidean")

# plot the distance matrix
# fviz_dist(dat.dis, gradient = list(low = "#b31b1b", mid = "white", high = "cornflowerblue")) + 
  ggtitle("\nAge & Fare Distance Matrix") + 
  theme_void(20) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())

# ---- k-Means Clustering -----------------------------------------------------------

{

# check out how the total within-cluster sum of squares and average silhouette width changes across different cluster sizes
stat.plots = lapply(c("wss", "silhouette"), function(i) 
  fviz_nbclust(dat.cluster, kmeans, nstart = 50, iter.max = 50, k.max = 10, method = i) + 
    geom_point(size = 2, color = "steelblue") + 
    geom_line(size = 1.5, color = "steelblue") + 
    ggtitle("k-Means Optimal Number of Clusters") +
    theme_bw(15) + 
    theme(plot.title = element_text(hjust = 0.5)))

# plot the wss and silhouette plots
# do.call(grid.arrange, c(stat.plots, nrow = 1))

# lets look at k cluster sizes
k = 6:9
set.seed(42)
mods = lapply(k, function(i) kmeans(dat.cluster, i, nstart = 50, iter.max = 50))

# build the cluster options
cluster.plots = lapply(1:length(mods), function(i) 
  fviz_cluster(mods[[i]], data = dat.cluster, stand = FALSE, geom = "point", pointsize = 2) + 
    ggtitle(paste0("k-Means ", k[i], " Cluster Plot")) +
    theme_bw(15) + 
    theme(plot.title = element_text(hjust = 0.5), legend.position = "top"))

# plot the cluster options
# do.call(grid.arrange, c(cluster.plots, nrow = 2))

# look at how the data is spread across clusters
par(mfrow = c(2, 2))
lapply(1:length(k), function(i) barplot(table(mods[[i]]$cluster), main = paste(k[i], "Clusters")))

# 8 clusters look good
kmeans.mod = mods[[which(k == 8)]]

# remove objects we no longer need
rm(stat.plots, k, mods, cluster.plots)

# free memory
gc()

}

# ---- Hierarchical k-Means Clustering ----------------------------------------------

{

# lets see which hierarchical clustering method best summarizes the data

# choice of hclust methods
methods = c("ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid")

# build cluster methods
hc = lapply(methods, function(i) hclust(dat.dis, method = i))

# compute cophenetic distance correlations
cors = rbindlist(lapply(1:length(methods), function(i) 
  data.table(method = methods[i], value = cor(dat.dis, cophenetic(hc[[i]])))))

# order by value
cors = cors[order(value, decreasing = TRUE)]

# make method a factor for plotting purposes
cors[, method := factor(method, levels = unique(method))]

# plot correlations
ggplot(cors, aes(x = method, y = value, fill = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "method", y = "correlation") +
  scale_fill_gradient(low = "yellow", high = "red") +
  theme_dark(15) +
  theme(legend.position = "none")

# lets go with average

# check out how the total within-cluster sum of squares and average silhouette width changes across different cluster sizes
stat.plots = lapply(c("wss", "silhouette"), function(i) 
  fviz_nbclust(dat.cluster, hkmeans, hc.method = "average", iter.max = 50, k.max = 10, method = i) + 
    geom_point(size = 2, color = "steelblue") + 
    geom_line(size = 1.5, color = "steelblue") + 
    ggtitle("Average Hierarchical k-Means Optimal Number of Clusters") +
    theme_bw(15) + 
    theme(plot.title = element_text(hjust = 0.5)))

# plot the wss and silhouette plots
# do.call(grid.arrange, c(stat.plots, nrow = 1))

# lets look at k cluster sizes
k = 6:9
set.seed(42)
mods = lapply(k, function(i) hkmeans(dat.cluster, i, hc.method = "average", iter.max = 50))

# build the cluster options
cluster.plots = lapply(1:length(mods), function(i) 
  fviz_cluster(mods[[i]], data = dat.cluster, stand = FALSE, geom = "point", pointsize = 2) + 
    ggtitle(paste0("Average Hierarchical k-Means ", k[i], " Cluster Plot")) +
    theme_bw(15) + 
    theme(plot.title = element_text(hjust = 0.5), legend.position = "top"))

# plot the cluster options
# do.call(grid.arrange, c(cluster.plots, nrow = 2))

# look at how the data is spread across clusters
par(mfrow = c(2, 2))
lapply(1:length(k), function(i) barplot(table(mods[[i]]$cluster), main = paste(k[i], "Clusters")))

# 9 clusters look good
hkmeans.average.mod = mods[[which(k == 9)]]

# plot the Dendrogram
# fviz_dend(hkmeans.average.mod, rect = TRUE, rect_border = "black", show_labels = FALSE) +
  ggtitle(paste("\nAverage Hierarchical k-Means", max(hkmeans.average.mod$cluster), "Cluster Dendrogram")) +		
  theme_void(15) + 
  theme(plot.title = element_text(hjust = 0.5))

# remove objects we no longer need
rm(stat.plots, k, mods, cluster.plots, cors, hc, methods)

# free memory
gc()

}

# ---- Cluster Selection ------------------------------------------------------------

{

# setup a list for the plots
cluster.plots = list()

# k-Means
cluster.plots[[length(cluster.plots) + 1]] = 
  fviz_cluster(kmeans.mod, data = dat.cluster, stand = FALSE, geom = "point", pointsize = 2) + 
  ggtitle("k-Means Cluster Plot") +
  theme_bw(15) + 
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")

# Average Hierarchical k-Means
cluster.plots[[length(cluster.plots) + 1]] = 
  fviz_cluster(hkmeans.average.mod, stand = FALSE, geom = "point", pointsize = 2) + 
  ggtitle("Average Hierarchical k-Means Cluster Plot") +
  theme_bw(15) + 
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")

# produce cluster plots
# do.call(grid.arrange, c(cluster.plots, nrow = 1))

# plot the distribtuion of data across clusters for each model
par(mfrow = c(1, 2))
ymax = nrow(dat.cluster)
barplot(table(kmeans.mod$cluster), main = "k-Means", ylim = c(0, ymax))
barplot(table(hkmeans.average.mod$cluster), main = "Average Hierarchical k-Means", ylim = c(0, ymax))

# lets go with k-means

# give dat the passenger clusters
dat[, Pcode := factor(kmeans.mod$cluster, levels = sort(unique(kmeans.mod$cluster)))]

# remove objects we no longer need
rm(cluster.plots, kmeans.mod, hkmeans.average.mod, dat.cluster, dat.dis, ymax)

# free memory
gc()

}

}

# ---- Scaling ----------------------------------------------------------------------

{

# remove the PassengerId column so we can scale the data
dat[, PassengerId := NULL]

# reformat all columns to be numeric by creating dummy variables for factor columns
dat = data.table(model.matrix(~., dat)[,-1])

# scale dat so that all variables can be compared fairly
dat = data.table(scale(dat))

# give the PassengerId column back to dat
dat[, PassengerId := 1:nrow(dat)]

# split up dat into train and test
train = cbind(Survived = train$Survived, dat[PassengerId %in% train$PassengerId])
test = dat[PassengerId %in% test$PassengerId]

# remove the PassengerId column from train and test
train[, PassengerId := NULL]
test[, PassengerId := NULL]

# remove objects we no longer need
rm(dat)

# free memory
gc()

}

}

# -----------------------------------------------------------------------------------
# ---- Feature Selection ------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- ANOVA ------------------------------------------------------------------------

{

# build a copy of train for aov
aov.dat = data.table(train[,!"Survived"])

# build all two-way interactions
aov.dat = data.table(model.matrix(~.^2, aov.dat)[,-1])

# create column names for aov.dat
# remove all `
aov.names = gsub("`", "", names(aov.dat))

# remove all spacing
aov.names = gsub(" ", "", aov.names)

# replace ":" with "."
aov.names = gsub(":", ".", aov.names)

# set the names of aov.dat
setnames(aov.dat, aov.names)

# attach Survived to aov.dat
aov.dat = cbind(Survived = as.numeric(as.character(train$Survived)), aov.dat)

# ---- Cut 1: Keep variables with p-value < 0.50 -----------------------------------

# build an anova table
my.aov = aov(Survived ~., data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.5
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.5, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# setup aov.dat to have the variables in keep.var
aov.dat = data.table(aov.dat[, c("Survived", keep.var), with = FALSE])

# ---- Cut 2: Keep variables with p-value < 0.25 -----------------------------------

# build an anova table
my.aov = aov(Survived ~., data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.5
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.25, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# setup aov.dat to have the variables in keep.var
aov.dat = data.table(aov.dat[, c("Survived", keep.var), with = FALSE])

# ---- Cut 3: Keep variables with p-value < 0.10 -----------------------------------

# build an anova table
my.aov = aov(Survived ~., data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.1
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.1, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# make sure only aov.names are in keep.var
keep.var = keep.var[which((keep.var %in% aov.names) == TRUE)]

# remove objects we no longer need
rm(aov.dat, my.aov, aov.names)

# free memory
gc()

}

# ---- Correlation ------------------------------------------------------------------

{

# build a copy of train for modeling
mod.dat = data.table(train)

# extract all potential variables
cor.dat = data.table(mod.dat[,!"Survived"])

# build all two-way interactions
cor.dat = data.table(model.matrix(~.^2, cor.dat)[,-1])

# create column names for cor.dat
# remove all `
cor.names = gsub("`", "", names(cor.dat))

# remove all spacing
cor.names = gsub(" ", "", cor.names)

# replace ":" with "."
cor.names = gsub(":", ".", cor.names)

# set the names of aov.dat
setnames(cor.dat, cor.names)

# attach cor.dat to mod.dat
mod.dat = cbind(Survived = mod.dat$Survived, cor.dat)

# setup mod.dat and cor.dat to have the variables in keep.var
mod.dat = data.table(mod.dat[, c("Survived", keep.var), with = FALSE])
cor.dat = data.table(cor.dat[, keep.var, with = FALSE])

# compute correlations
cors = cor(cor.dat)
# replace any NA's with 1's
cors[is.na(cors)] = 1

# find out which variables are highly correlated (>= 0.9) and remove them
find.dat = findCorrelation(cors, cutoff = 0.9, names = TRUE)

# remove columns from mod.dat according to find.dat
if(length(find.dat) > 0) mod.dat = mod.dat[, !find.dat, with = FALSE]

}

# ---- Importance -------------------------------------------------------------------

{

# lets use random forests to determine variable importance
# the classes are imbalanced so lets define the sampsize parameter
sampsize = table(train$Survived)
sampsize = ceiling(.632 * rep(min(sampsize), length(sampsize)))

# make sure to randomly sample within each class of Survived
strata = train$Survived

# choose the number of workers and tasks for parallel processing
workers = 10
tasks = 10

# set up seeds for reproducability
set.seed(42)
seeds = sample(1:1000, tasks)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("random forest - variable importance\n")
cat(paste(workers, "workers started at", Sys.time()), "\n")
sink()

# build random forest models in parallel
var.imp = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(randomForest)
  require(data.table)
  require(caret)
  
  # build randomForest
  set.seed(seeds[i])
  mod = randomForest(Survived ~ ., data = mod.dat, 
                     sampsize = sampsize, strata = strata, 
                     ntree = 1000, importance = TRUE)
  
  # compute variable importance on a scale of 0 to 100
  imp = varImp(mod, scale = TRUE)
  
  # transform imp into long format
  imp = data.table(variable = rownames(imp), 
                   value = (rowSums(imp) / ncol(imp)) / 100)
  
  # add the task number to imp
  imp[, task := i]
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time()), "\n")
  sink()
  
  return(imp)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time()), "\n")
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of data tables into one table
var.imp = rbindlist(var.imp)

# average importance of variables
var.imp = var.imp[, .(value = mean(value)), by = .(variable)]

# order by importance
var.imp = var.imp[order(value, decreasing = TRUE)]

# make variable a factor for plotting purposes
var.imp[, variable := factor(variable, levels = unique(variable))]

# plot a barplot of variable importance
ggplot(var.imp, aes(x = variable, y = value, fill = value, color = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Variable", y = "Importance") +
  scale_y_continuous(labels = percent) +
  scale_fill_gradient(low = "yellow", high = "red") +
  scale_color_gradient(low = "yellow", high = "red") +
  theme_dark(15) +
  theme(legend.position = "none", axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major.x = element_blank())

# lets only keep variables with at least 5% importance
keep.dat = gsub("`", "", var.imp[value >= 0.05, variable])
mod.dat = mod.dat[, c("Survived", keep.dat), with = FALSE]

# heres our variables for Survived
Survived.variables = keep.dat

}

# ---- Finalize Data ----------------------------------------------------------------

{

# setup train to have the variables in SalePrice.variables
train = data.table(mod.dat)

# setup test to have the variables in Survived.variables
# build all two-way interactions
test = data.table(model.matrix(~.^2, test))

# create column names for test
# remove all `
test.names = gsub("`", "", names(test))

# remove all spacing
test.names = gsub(" ", "", test.names)

# rename column names to have "." instead of ":"
test.names = gsub(":", ".", test.names)

# set the names of test
setnames(test, test.names)

# only keep the model variables
test = test[, Survived.variables, with = FALSE]

# remove objects we no longer need
rm(strata, seeds, var.imp, tasks, workers, cl, sampsize, test.names, cor.names, keep.var, Survived.variables, cor.dat, cors, find.dat, mod.dat, keep.dat)

# free memory
gc()

}

}

# -----------------------------------------------------------------------------------
# ---- Logistic Regression Model ----------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# extract predictors (X) and response (Y)
X = data.table(train[,!"Survived"])
Y = train$Survived

# build the cross validation folds
cv = cv.folds(n = nrow(X), K = K, R = R, seed = 42)

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our models
log.pred = function(Xtrain, Ytrain, Xtest, Ytest)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  set.seed(42)
  mod = glm(y ~ ., data = dat, 
            family = binomial(link = "logit"), 
            control = list(maxit = 100))
  
  # make predictions with the training model using the test set
  ynew = predict(mod, data.table(Xtest), type = "response")
  Ytest = as.numeric(as.character(Ytest))
  
  # compute the cut-off point that maximize accuracy
  mod.roc = roc(Ytest ~ ynew)
  cutoff = coords(mod.roc, x = "best", best.weights = c(1, cases / population))[1]
  
  # use the cutoff point to define predictions
  ynew = as.numeric(ynew >= cutoff)
  
  # compute a binary confusion matrix
  conf = confusion(ytrue = Ytest, ypred = ynew)
  
  # extract the four cases from conf
  conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
  
  # build a table to summarize the performance of our training model
  output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                       Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                       Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                       AUC = as.numeric(auc(mod.roc)),
                       Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                       Cutoff = cutoff)
  
  # replace any NaN with NA
  output = as.matrix(output)
  output[is.nan(output)] = NA
  output = data.table(output)
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = length(cv)
tasks = length(cv)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("logistic regression - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation
log.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(pROC)
  
  # extract the training and test sets
  folds = cv[[i]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = log.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
log.cv = rbindlist(log.cv)

# summarize performance metrics for every model
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

log.diag = log.cv[,.(stat = factor(stat, levels = stat),
                     Accuracy = as.vector(summary(na.omit(Accuracy))), 
                     Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                     Specificity = as.vector(summary(na.omit(Specificity))),
                     AUC = as.vector(summary(na.omit(AUC))),
                     Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                     Cutoff = as.vector(summary(na.omit(Cutoff))))]

# ---- Results ----------------------------------------------------------------------

# add a model name column
log.diag[, mod := rep("log", nrow(log.diag))]

# store model diagnostic results
mods.diag = data.table(log.diag)

# build the model
set.seed(42)
log.mod = glm(Survived ~ ., data = train, 
              family = binomial(link = "logit"), 
              control = list(maxit = 100))

# store the model
log.list = list("mod" = log.mod)
mods.list = list("log" = log.list)

# remove objects we no longer need
rm(log.cv, log.diag, log.list, log.mod, X, Y, cl, stat, tasks, workers, log.pred)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Penalty Regression Model -----------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# extract predictors (X) and response (Y)
X = as.matrix(train[,!"Survived"])
Y = train$Survived

# glmnet offers:
# ridge penalty by setting the parameter alpha = 0
# lasso penalty by setting the parameter alpha = 1
# elastic net penalty by setting the parameter 0 < alpha < 1

# build a sequence of alpha values to test
doe = data.table(alpha = seq(0, 1, 0.05))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our models
pen.pred = function(Xtrain, Ytrain, Xtest, Ytest, alpha)
{
  # build the training model
  set.seed(42)
  mod = cv.glmnet(x = Xtrain, y = Ytrain, family = "binomial", alpha = alpha)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, s = mod$lambda.min, Xtest, type = "response"))
  Ytest = as.numeric(as.character(Ytest))
  
  # compute the cut-off point that maximize accuracy
  mod.roc = roc(Ytest ~ ynew)
  cutoff = coords(mod.roc, x = "best", best.weights = c(1, cases / population))[1]
  
  # use the cutoff point to define predictions
  ynew = as.numeric(ynew >= cutoff)
  
  # compute a binary confusion matrix
  conf = confusion(ytrue = Ytest, ypred = ynew)
  
  # extract the four cases from conf
  conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
  
  # build a table to summarize the performance of our training model
  output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                      Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                      Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                      AUC = as.numeric(auc(mod.roc)),
                      Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                      Cutoff = cutoff)
  
  # replace any NaN with NA
  output = as.matrix(output)
  output[is.nan(output)] = NA
  output = data.table(output)
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 15
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("penalty regression - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
pen.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(glmnet)
  require(pROC)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = pen.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, alpha = doe$alpha[i])
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
pen.cv = rbindlist(pen.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

pen.diag = pen.cv[,.(stat = factor(stat, levels = stat),
                     Accuracy = as.vector(summary(na.omit(Accuracy))), 
                     Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                     Specificity = as.vector(summary(na.omit(Specificity))),
                     AUC = as.vector(summary(na.omit(AUC))),
                     Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                     Cutoff = as.vector(summary(na.omit(Cutoff)))),
                  by = alpha]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(pen.diag)
pen.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert pen.diag into long format for plotting purposes
DT = data.table(melt(pen.diag, measure.vars = c("Accuracy", "Sensitivity", "Specificity", "AUC", "Odds.Ratio", "Cutoff")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter out models
pen.diag[stat == "Median" & Odds.Ratio >= 32 & AUC >= 0.88]

# model 1 looks good
set.seed(42)
pen.mod = cv.glmnet(x = X, y = Y, family = "binomial", alpha = 0)

# extract coefficients of the chosen terms for the lambda that minimizes mean cross-validated error
pen.coef = coef(pen.mod, s = "lambda.min")
pen.coef = data.table(term = rownames(pen.coef), coefficient = as.numeric(pen.coef))

# store model diagnostic results
pen.diag = pen.diag[mod == 1]
pen.diag[, mod := rep("pen", nrow(pen.diag))]
pen.diag = pen.diag[,.(Accuracy, Sensitivity, Specificity, AUC, Odds.Ratio, Cutoff, stat, mod)]
mods.diag = rbind(mods.diag, pen.diag)

# store the model
pen.list = list("mod" = pen.mod, "coef" = pen.coef)
mods.list$pen = pen.list

# remove objects we no longer need
rm(pen.cv, pen.diag, pen.list, pen.mod, pen.pred, doe, DT, 
   pen.coef, X, Y, diag.plot, cl, workers, tasks, num.stats, num.rows, stat)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Gradient Boosting Model ------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# we have 7 hyperparameters of interest:
  # nrounds ~ the max number of boosting iterations
  # eta ~ the learning rate
  # max_depth ~ maximum depth of a tree
  # min_child_weight ~ minimum sum of instance weight needed in a child
  # gamma ~ minimum loss reduction required to make a further partition on a leaf node of the tree
  # subsample ~ the proportion of data (rows) to randomly sample each round
  # colsample_bytree ~ the proportion of variables (columns) to randomly sample each round

# check out this link for help on tuning:
  # https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur
  # google stuff and you'll find other approaches

# extract predictors (X) and response (Y)
X = as.matrix(train[,!"Survived"])
Y = as.numeric(as.character(train$Survived))

# create parameter combinations to test
doe = data.table(expand.grid(nrounds = 100,
                             eta = 0.1,
                             max_depth = c(4, 6, 8, 10, 12, 14), 
                             min_child_weight = c(1, 3, 5, 7, 9, 11),
                             gamma = 0,
                             subsample = 1,
                             colsample_bytree = 1))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# the classes are imbalanced so lets define the scale.pos.weight parameter: sum(negative cases) / sum(positive cases)
scale.pos.weight = (population - cases) / cases

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
gbm.pred = function(Xtrain, Ytrain, Xtest, Ytest, objective, eval_metric, eta, max_depth, nrounds, min_child_weight, gamma, subsample, colsample_bytree)
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Ytrain, data = Xtrain,
                objective = objective, eval_metric = eval_metric,
                eta = eta, max_depth = max_depth,
                nrounds = nrounds, min_child_weight = min_child_weight,
                gamma = gamma, verbose = 0, scale.pos.weight = scale.pos.weight,
                subsample = subsample, colsample_bytree = colsample_bytree)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = Xtest))
  
  # compute the cut-off point that maximize accuracy
  mod.roc = roc(Ytest ~ ynew)
  cutoff = coords(mod.roc, x = "best", best.weights = c(1, cases / population))[1]
  
  # use the cutoff point to define predictions
  ynew = as.numeric(ynew >= cutoff)
  
  # compute a binary confusion matrix
  conf = confusion(ytrue = Ytest, ypred = ynew)
  
  # extract the four cases from conf
  conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
  
  # build a table to summarize the performance of our training model
  output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                      Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                      Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                      AUC = as.numeric(auc(mod.roc)),
                      Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                      Cutoff = cutoff)
  
  # replace any NaN with NA
  output = as.matrix(output)
  output[is.nan(output)] = NA
  output = data.table(output)
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 6
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("gradient boosting - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
gbm.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(xgboost)
  require(pROC)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = gbm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    objective = "binary:logistic", eval_metric = "error",
                    eta = doe$eta[i], max_depth = doe$max_depth[i], nrounds = doe$nrounds[i], 
                    min_child_weight = doe$min_child_weight[i], gamma = doe$gamma[i], 
                    subsample = doe$subsample[i], colsample_bytree = doe$colsample_bytree[i])
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
gbm.cv = rbindlist(gbm.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

gbm.diag = gbm.cv[,.(stat = factor(stat, levels = stat),
                     Accuracy = as.vector(summary(na.omit(Accuracy))), 
                     Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                     Specificity = as.vector(summary(na.omit(Specificity))),
                     AUC = as.vector(summary(na.omit(AUC))),
                     Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                     Cutoff = as.vector(summary(na.omit(Cutoff)))),
                  by = .(eta, max_depth, nrounds, min_child_weight, gamma, subsample, colsample_bytree)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(gbm.diag)
gbm.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert gbm.diag into long format for plotting purposes
DT = data.table(melt(gbm.diag, measure.vars = c("Accuracy", "Sensitivity", "Specificity", "AUC", "Odds.Ratio", "Cutoff")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
gbm.diag[stat == "Median" & AUC >= 0.88 & Odds.Ratio >= 27 & mod %in% gbm.diag[stat == "Min" & Odds.Ratio >= 20, mod]]

# model 25 looks good
gbm.diag = gbm.diag[mod == 25]

# rename model to gbm
gbm.diag[, mod := rep("gbm", nrow(gbm.diag))]

# recall scale.pos.weight
scale.pos.weight

# build our model
set.seed(42)
gbm.mod = xgboost(label = Y, data = X,
                  objective = "binary:logistic", eval_metric = "error",
                  eta = 0.1, max_depth = 4,
                  nrounds = 100, min_child_weight = 9,
                  gamma = 0, verbose = 0, scale.pos.weight = 1.605263,
                  subsample = 1, colsample_bytree = 1)

# store model diagnostic results
gbm.diag = gbm.diag[,.(Accuracy, Sensitivity, Specificity, AUC, Odds.Ratio, Cutoff, stat, mod)]
mods.diag = rbind(mods.diag, gbm.diag)

# store the model
gbm.list = list("mod" = gbm.mod)
mods.list$gbm = gbm.list

# remove objects we no longer need
rm(gbm.cv, gbm.diag, gbm.list, gbm.mod, gbm.pred, doe, DT, scale.pos.weight,
   X, Y, diag.plot, cl, workers, tasks, num.stats, num.rows, stat)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Random Forest Model ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# we have 2 hyperparameters of interest:
  # ntree ~ number of decision trees to create
  # nodesize ~ minimum size of terminal nodes (ie. the minimum number of data points that can be grouped together in any node of a tree)

# check out this link for help on tuning:
# https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur

# extract predictors (X) and response (Y)
X = data.table(train[,!"Survived"])
Y = train$Survived

# create parameter combinations to test
doe = data.table(expand.grid(ntree = c(500, 800, 1200), 
                             nodesize = c(1, 3, 5, 10)))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# the classes are imbalanced so lets define the sampsize and strata parameters
sampsize = table(Y)
sampsize = ceiling(.632 * rep(min(sampsize), length(sampsize)))
strata = Y

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
rf.pred = function(Xtrain, Ytrain, Xtest, Ytest, ntree, nodesize, sampsize, strata)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  set.seed(42)
  mod = randomForest(y ~ .,
                     data = dat,
                     ntree = ntree,
                     nodesize = nodesize,
                     sampsize = sampsize,
                     strata = strata)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = data.table(Xtest), type = "prob")[,2])
  Ytest = as.numeric(as.character(Ytest))
  
  # compute the cut-off point that maximize accuracy
  mod.roc = roc(Ytest ~ ynew)
  cutoff = coords(mod.roc, x = "best", best.weights = c(1, cases / population))[1]
  
  # use the cutoff point to define predictions
  ynew = as.numeric(ynew >= cutoff)
  
  # compute a binary confusion matrix
  conf = confusion(ytrue = Ytest, ypred = ynew)
  
  # extract the four cases from conf
  conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
  
  # build a table to summarize the performance of our training model
  output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                      Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                      Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                      AUC = as.numeric(auc(mod.roc)),
                      Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                      Cutoff = cutoff)
  
  # replace any NaN with NA
  output = as.matrix(output)
  output[is.nan(output)] = NA
  output = data.table(output)
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 15
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("random forest - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
rf.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(randomForest)
  require(pROC)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # update sampsize due to Xtrain having less data than X
  sampsize.update = round((nrow(Xtrain) / nrow(X)) * sampsize, 0)
  
  # build model and get prediction results
  output = rf.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                   ntree = doe$ntree[i], nodesize = doe$nodesize[i],
                   sampsize = sampsize.update, strata = strata)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
rf.cv = rbindlist(rf.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

rf.diag = rf.cv[,.(stat = factor(stat, levels = stat),
                     Accuracy = as.vector(summary(na.omit(Accuracy))), 
                     Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                     Specificity = as.vector(summary(na.omit(Specificity))),
                     AUC = as.vector(summary(na.omit(AUC))),
                     Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                     Cutoff = as.vector(summary(na.omit(Cutoff)))),
                  by = .(ntree, nodesize)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(rf.diag)
rf.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert rf.diag into long format for plotting purposes
DT = data.table(melt(rf.diag, measure.vars = c("Accuracy", "Sensitivity", "Specificity", "AUC", "Odds.Ratio", "Cutoff")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
rf.diag[stat == "Median" & Odds.Ratio >= 30]

# model 6 looks good
rf.diag = rf.diag[mod == 6]

# rename model to rf
rf.diag[, mod := rep("rf", nrow(rf.diag))]

# recall sampsize
sampsize

# build the model
set.seed(42)
rf.mod = randomForest(Survived ~ ., data = train, ntree = 1200, 
                      nodesize = 3, sampsize = c(217, 217), strata = train$Survived)

# store model diagnostic results
rf.diag = rf.diag[, .(Accuracy, Sensitivity, Specificity, AUC, Odds.Ratio, Cutoff, stat, mod)]
mods.diag = rbind(mods.diag, rf.diag)

# store the model
rf.list = list("mod" = rf.mod)
mods.list$rf = rf.list

# remove objects we no longer need
rm(rf.cv, rf.diag, rf.list, rf.mod, rf.pred, doe, DT, X, Y, diag.plot,
   cl, workers, tasks, num.stats, num.rows, stat, sampsize, strata)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Deep Nueral Network Model ----------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# extract predictors (X) and response (Y)
X = data.table(train[,!"Survived"])
Y = train$Survived

# check out the following link to understand h2o deep learning
# http://h2o-release.s3.amazonaws.com/h2o/rel-tukey/6/docs-website/h2o-docs/booklets/R_Vignette.pdf

# we have 3 hyperparameters of interest:
# hidden ~ a vector of integers indicating the number of nodes in each hidden layer
# l1 ~ L1 norm regularization to penalize large weights (may cause many weights to become 0)
# l2 ~ L2 norm regularization to penalize large weights (may cause many weights to become small)

# set up L1 & L2 penalties
l1 = 1e-5
l2 = 1e-5

# use the same seed that we've been using for model building
seed = 42

# how many times the training data should be passed through the network to adjust path weights
epochs = 50

# the classes are imbalanced so lets set up the balance_classes and class_sampling_factors parameters
balance_classes = TRUE
class_sampling_factors = table(Y)
class_sampling_factors = as.vector(max(class_sampling_factors) / class_sampling_factors)

# choose the total number of hidden nodes
nodes = 150

# choose the hidden layer options to distribtuion nodes across
layers = 1:5

# choose whether to try varying structures for each layer (0 = No, 1 = Yes)
vary = 0

# initilize the size of doe
N = max(layers)
doe = matrix(ncol = N)

# build different ratios for distributing nodes across hidden layer options
for(n in layers)
{
  # single layer
  if(n == 1)
  {
    # just one layer
    op = c(1, rep(0, N - n))
    
    # store layer option
    doe = rbind(doe, op)
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op)
    
    # double layer
  } else if(n == 2)
  {
    # layers increase in size
    op1 = c(1:n, rep(0, N - n))
    # layers decrease in size
    op2 = c(n:1, rep(0, N - n))
    # layers are equal in size
    op3 = c(rep(1, length.out = n), rep(0, N - n))
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3)
    
    # largest multi-layer
  } else if(n == N)
  {
    # layers increase in size
    op1 = 1:n
    # layers decrease in size
    op2 = n:1
    # layers are equal in size
    op3 = rep(1, length.out = n)
    # layers oscilate in size, starting low
    op4 = rep(1:2, length.out = n)
    # layers oscilate in size, starting high
    op5 = rep(2:1, length.out = n)
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    op4 = op4 / sum(op4)
    op5 = op5 / sum(op5)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3, op4, op5)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3, op4, op5)
    
    # not the largest multi-layer
  } else
  {
    # op1 through op5 are the same as above
    op1 = c(1:n, rep(0, N - n))
    op2 = c(n:1, rep(0, N - n))
    op3 = c(rep(1, length.out = n), rep(0, N - n))
    op4 = c(rep(1:2, length.out = n), rep(0, N - n))
    op5 = c(rep(2:1, length.out = n), rep(0, N - n))
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    op4 = op4 / sum(op4)
    op5 = op5 / sum(op5)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3, op4, op5)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3, op4, op5)
  }
}

rm(n, N)

# remove the first row of doe becuase it was just a dummy row to append to
doe = doe[-1,]
doe = data.frame(doe)

# add cross validation ids for each scenario in doe
doe = data.frame(rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe))))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
dnn.pred = function(Xtrain, Ytrain, Xtest, Ytest, hidden, l1, l2, epochs, seed, balance_classes, class_sampling_factors)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # make dat and Xtest into h2o objects
  dat.h2o = as.h2o(dat)
  Xtest.h2o = as.h2o(Xtest)
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         hidden = hidden,
                         l1 = l1,
                         l2 = l2,
                         epochs = epochs,
                         seed = seed,
                         balance_classes = balance_classes,
                         class_sampling_factors = class_sampling_factors,
                         # activation = "Tanh",
                         # max_w2 = 10,
                         # initial_weight_distribution = "UniformAdaptive",
                         # initial_weight_scale = 0.5,
                         variable_importances = FALSE)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
  Ytest = as.numeric(as.character(Ytest))
  
  # compute the cut-off point that maximize accuracy
  mod.roc = roc(Ytest ~ ynew)
  cutoff = coords(mod.roc, x = "best", best.weights = c(1, cases / population))[1]
  
  # compute a binary confusion matrix
  conf = confusion(ytrue = Ytest, ypred = ynew)
  
  # extract the four cases from conf
  conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
  
  # build a table to summarize the performance of our training model
  output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                      Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                      Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                      AUC = as.numeric(auc(mod.roc)),
                      Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                      Cutoff = cutoff)
  
  # replace any NaN with NA
  output = as.matrix(output)
  output[is.nan(output)] = NA
  output = data.table(output)
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of tasks
tasks = nrow(doe)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("deep nueral network - cross validation\n")
cat(paste("task 1 started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
dnn.cv = foreach(i = 1:tasks) %do%
{
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # extract the portion of doe regarding the hidden layer structure
  doe.size = doe[,-1]
  
  # build the hidden layer structure for model i
  size = length(which(doe.size[i,] > 0))
  hidden = sapply(1:size, function(j) round(ceiling(nodes * doe.size[i,j]), 0))
  
  # build model and get prediction results
  output = dnn.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    hidden = hidden, l1 = l1, l2 = l2, epochs = epochs, seed = seed, 
                    balance_classes = balance_classes, class_sampling_factors = class_sampling_factors)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i,])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# free memory
gc()

# combine the list of tables into one table
dnn.cv = rbindlist(dnn.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

dnn.diag = dnn.cv[,.(stat = factor(stat, levels = stat),
                     Accuracy = as.vector(summary(na.omit(Accuracy))), 
                     Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                     Specificity = as.vector(summary(na.omit(Specificity))),
                     AUC = as.vector(summary(na.omit(AUC))),
                     Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                     Cutoff = as.vector(summary(na.omit(Cutoff)))),
                  by = eval(paste0("X", seq(1:(ncol(doe) - 1))))]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(dnn.diag)
dnn.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert dnn.diag into long format for plotting purposes
DT = data.table(melt(dnn.diag, measure.vars = c("Accuracy", "Sensitivity", "Specificity", "AUC", "Odds.Ratio", "Cutoff")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
dnn.diag[stat == "Median" & Accuracy >= 0.82 & mod %in% dnn.diag[stat == "Min" & Odds.Ratio >= 10, mod]]

# model 10 looks good
dnn.diag = dnn.diag[mod == 10]

# rename model to dnn
dnn.diag[, mod := rep("dnn", nrow(dnn.diag))]

# build the hidden layer structure for model i
i = 10
doe.size = doe[,-1]
size = length(which(doe.size[i,] > 0))
hidden = sapply(1:size, function(j) round(ceiling(nodes * doe.size[i,j]), 0))
hidden

# recall the other parameters
l1
l2
epochs
class_sampling_factors
seed

# build the model
train.h2o = as.h2o(train)
dnn.mod = h2o.deeplearning(y = "Survived",
                           x = colnames(X),
                           training_frame = train.h2o,
                           hidden = c(115, 96, 77, 58, 39, 20),
                           l1 = 1e-05,
                           l2 = 1e-05,
                           epochs = 100,
                           seed = 42,
                           balance_classes = TRUE,
                           class_sampling_factors = c(1, 1.605263),
                           variable_importances = FALSE)

# store model diagnostic results
dnn.diag = dnn.diag[,.(Accuracy, Sensitivity, Specificity, AUC, Odds.Ratio, Cutoff, stat, mod)]
mods.diag = rbind(mods.diag, dnn.diag)

# store the model
dnn.list = list("mod" = dnn.mod)
mods.list$dnn = dnn.list

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# remove objects we no longer need
rm(dnn.cv, dnn.diag, dnn.list, dnn.mod, dnn.pred, i, doe, DT, output, X, Y, diag.plot, 
   layers, seed, train.h2o, epochs, hidden, l1 ,l2, nodes, size, balance_classes, class_sampling_factors,
   doe.size, Xtest, Xtrain, num.rows, num.stats, stat, tasks, Ytest, Ytrain, folds)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Support Vector Machine Model -------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# hyperparameters of interest:
# cost ~ controls the error penalty for misclassification
  # higher values increase the error penalty and decrease the margin of seperation
  # lower values decrease the error penalty and increase the margin of seperation
# gamma ~ controls the radius of the region of influence for support vectors
  # if too large, the region of influence of any selected support vectors would only include the support vector itself and overfit the data.
  # if too small, the region of influence of any selected support vector would include the whole training set and underfit the data

# check out this link for help on tuning:
# https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur

# default value for gamma in our case:
gamma = 1 / ncol(train[,!"Survived"])
gamma

# extract predictors (X) and response (Y)
X = data.table(train[,!"Survived"])
Y = train$Survived

# create parameter combinations to test
doe = data.table(expand.grid(cost = lseq(0.001, 1000, 500),
                             gamma = gamma))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# the classes are imbalanced so lets define the class.weights parameter
class.weights = table(Y)
class.weights = setNames(as.vector(max(class.weights) / class.weights), names(class.weights))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
svm.pred = function(Xtrain, Ytrain, Xtest, Ytest, cost, gamma, class.weights)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  set.seed(42)
  mod = svm(y ~ .,
            data = dat,
            cost = cost,
            gamma = gamma,
            class.weights = class.weights)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(predict(mod, newdata = data.table(Xtest))))
  Ytest = as.numeric(as.character(Ytest))
  
  # compute the cut-off point that maximize accuracy
  mod.roc = roc(Ytest ~ ynew)
  cutoff = coords(mod.roc, x = "best", best.weights = c(1, cases / population))[1]
  
  # use the cutoff point to define predictions
  ynew = as.numeric(ynew >= cutoff)
  
  # compute a binary confusion matrix
  conf = confusion(ytrue = Ytest, ypred = ynew)
  
  # extract the four cases from conf
  conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
  
  # build a table to summarize the performance of our training model
  output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                      Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                      Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                      AUC = as.numeric(auc(mod.roc)),
                      Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                      Cutoff = cutoff)
  
  # replace any NaN with NA
  output = as.matrix(output)
  output[is.nan(output)] = NA
  output = data.table(output)
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 15
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("support vector machine - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
svm.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(e1071)
  require(pROC)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = svm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    cost = doe$cost[i], gamma = doe$gamma[i], class.weights = class.weights)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
svm.cv = rbindlist(svm.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

svm.diag = svm.cv[,.(stat = factor(stat, levels = stat),
                     Accuracy = as.vector(summary(na.omit(Accuracy))), 
                     Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                     Specificity = as.vector(summary(na.omit(Specificity))),
                     AUC = as.vector(summary(na.omit(AUC))),
                     Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                     Cutoff = as.vector(summary(na.omit(Cutoff)))),
                  by = .(cost, gamma)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(svm.diag)
svm.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert svm.diag into long format for plotting purposes
DT = data.table(melt(svm.diag, measure.vars = c("Accuracy", "Sensitivity", "Specificity", "AUC", "Odds.Ratio", "Cutoff")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter out models
svm.diag[stat == "Median" & Accuracy >= 0.82 & AUC >= 0.82 & Sensitivity >= 0.8 & mod %in% svm.diag[stat == "Min" & Odds.Ratio >= 0.15, mod]]

# lets go with model 279
svm.diag = svm.diag[mod == 279]

# rename model to svm as this is our chosen model
svm.diag[, mod := rep("svm", nrow(svm.diag))]

# recall class.weights
class.weights

# build our model
set.seed(42)
svm.mod = svm(Survived ~ ., data = train, cost = 2.201331, gamma = 1/45, class.weights = c("0" = 0.6229508, "1" = 1))

# store model diagnostic results
svm.diag = svm.diag[,.(Accuracy, Sensitivity, Specificity, AUC, Odds.Ratio, Cutoff, stat, mod)]
mods.diag = rbind(mods.diag, svm.diag)

# store the model
svm.list = list("mod" = svm.mod)
mods.list$svm = svm.list

# remove objects we no longer need
rm(gamma, svm.cv, svm.diag, svm.list, svm.mod, svm.pred, doe, DT, X, Y, diag.plot,
   workers, tasks, stat, num.stats, num.rows, class.weights, cl)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Super Learner Model ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# extract predictors (X) and response (Y)
sl.X = as.matrix(train[,!"Survived"])
sl.Y = as.numeric(as.character(train$Survived))

# create Super Learner wrappers for our models of interest

# logistic regression wrapper
my.log = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # build the training model
  set.seed(42)
  mod = glm(y ~ ., data = dat,
            family = binomial(link = "logit"), 
            control = list(maxit = 100))
  
  # make predictions with the training model using the test set
  ynew = predict(mod, data.table(newX), type = "response")
  ynew = as.numeric(ynew >= mods.diag[mod == "log" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# penalty regression wrapper
my.pen = function(Y, X, newX, ...)
{
  # make Y into a factor data type
  Y = factor(Y, levels = 0:1)
  
  # build the training model
  set.seed(42)
  mod = cv.glmnet(x = X, y = Y, family = "binomial", alpha = 0)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, s = mod$lambda.min, newX, type = "response"))
  ynew = as.numeric(ynew >= mods.diag[mod == "pen" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# gradient boosting wrapper
my.gbm = function(Y, X, newX, ...)
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Y, data = X,
                objective = "binary:logistic", eval_metric = "error",
                eta = 0.1, max_depth = 4,
                nrounds = 100, min_child_weight = 9,
                gamma = 0, verbose = 0, scale.pos.weight = 1.605263,
                subsample = 1, colsample_bytree = 1)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = newX))
  ynew = as.numeric(ynew >= mods.diag[mod == "gbm" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# random forest wrapper
my.rf = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # update sampsize
  sampsize = c(217, 217)
  sampsize.update = (nrow(dat) / nrow(sl.X)) * sampsize
  
  # build the training model
  set.seed(42)
  mod = randomForest(y ~ .,
                     data = dat,
                     ntree = 1200, 
                     nodesize = 3, 
                     sampsize = sampsize.update, 
                     strata = dat$y)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = data.table(newX), type = "prob")[,2])
  ynew = as.numeric(ynew >= mods.diag[mod == "rf" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# deep neural network wrapper
my.dnn = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # make dat and newX into h2o objects
  dat.h2o = as.h2o(dat)
  newX.h2o = as.h2o(data.table(newX))
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         hidden = c(115, 96, 77, 58, 39, 20),
                         l1 = 1e-05,
                         l2 = 1e-05,
                         epochs = 10,
                         seed = 42,
                         balance_classes = TRUE,
                         class_sampling_factors = c(1, 1.605263),
                         variable_importances = FALSE)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = newX.h2o))$predict))
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# support vector machine wrapper
my.svm = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # build the training model
  set.seed(42)
  mod = svm(y ~ .,
            data = dat,
            cost = 2.201331, 
            gamma = 1/45, 
            class.weights = c("0" = 0.6229508, "1" = 1))
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(predict(mod, newdata = data.table(newX))))
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# create a library of the above wrappers
my.library = list("my.log", "my.pen", "my.gbm", "my.rf", "my.dnn", "my.svm")

# ---- Choosing Models --------------------------------------------------------------

# build the super learner model
set.seed(42)
sl.mod = SuperLearner(Y = sl.Y, X = sl.X, family = binomial(), verbose = TRUE, 
                      SL.library = my.library, cvControl = list(stratifyCV = TRUE))
sl.mod

# all of the risks and coefficients are fine
# but having log and pen togther doesn't make sense as they both use least squarea QR decomposition to solve for variable coefficients
# in this case pen outperforms log, so lets remove log
my.library = my.library[-which(my.library %in% c("my.log"))]

# if dnn was removed then shutdown the h2o instance
if(!("my.dnn" %in% my.library))
{
  h2o.shutdown(prompt = FALSE)
}

# build the super learner model
set.seed(42)
sl.mod = SuperLearner(Y = sl.Y, X = sl.X, family = binomial(), verbose = TRUE, 
                      SL.library = my.library, cvControl = list(stratifyCV = TRUE))
sl.mod

# ---- CV ---------------------------------------------------------------------------

# the snow cluster won't work for SuperLearner
# even the example page won't run properly
# so we will have to do this cross validation sequentially
# thats why all of the parallel processing related commands are commented out
# also if the dnn is in the super learner than that is already using up a significant portion of the CPU
# so this cross validation shouldn't even be considered for parallel processing unless the dnn is not in my.library

# build a function that will report prediction results of our models
sl.pred = function(Xtrain, Ytrain, Xtest, Ytest, my.library)
{
  # build the training model
  set.seed(42)
  mod = SuperLearner(Y = Ytrain, X = Xtrain, newX = Xtest, 
                     family = binomial(), SL.library = my.library, 
                     cvControl = list(stratifyCV = TRUE))
  
  # make predictions with the training model using the test set
  ynew = as.numeric(mod$SL.predict)
  
  # compute the cut-off point that maximize accuracy
  mod.roc = roc(Ytest ~ ynew)
  cutoff = coords(mod.roc, x = "best", best.weights = c(1, cases / population))[1]
  
  # use the cutoff point to define predictions
  ynew = as.numeric(ynew >= cutoff)
  
  # compute a binary confusion matrix
  conf = confusion(ytrue = Ytest, ypred = ynew)
  
  # extract the four cases from conf
  conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
  
  # build a table to summarize the performance of our training model
  output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                      Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                      Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                      AUC = as.numeric(auc(mod.roc)),
                      Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                      Cutoff = cutoff)
  
  # replace any NaN with NA
  output = as.matrix(output)
  output[is.nan(output)] = NA
  output = data.table(output)
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
# workers = 5
tasks = length(cv)

# setup seeds for parallel processing
# set.seed(42)
# seeds = sample(1:1000, 6)

# setup parallel processing
# cl = makeCluster(workers, type = "SOCK", outfile = "")
# clusterSetupRNGstream(cl, seed = seeds)
# registerDoSNOW(cl)

# assign the prediction functions in my.library to the SuperLearner namespace
# environment(my.log) = asNamespace("SuperLearner")
# environment(my.pen) = asNamespace("SuperLearner")
# environment(my.gbm) = asNamespace("SuperLearner")
# environment(my.rf) = asNamespace("SuperLearner")
# environment(my.dnn) = asNamespace("SuperLearner")
# environment(my.svm) = asNamespace("SuperLearner")

# copy the prediction functions in my.library to all clusters
# clusterExport(cl, varlist = my.library)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("super learner - cross validation\n")
cat(paste("task 1 started at", Sys.time(), "\n"))
sink()

# perform cross validation
sl.cv = foreach(i = 1:tasks) %do%
{
  # extract the training and test sets
  folds = cv[[i]]
  Xtrain = sl.X[folds$train,]
  Ytrain = sl.Y[folds$train]
  Xtest= sl.X[folds$test,]
  Ytest = sl.Y[folds$test]
  
  # build model and get prediction results
  output = sl.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, my.library = my.library)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
# stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
sl.cv = rbindlist(sl.cv)

# summarize performance metrics for every model
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

sl.diag = sl.cv[,.(stat = factor(stat, levels = stat),
                     Accuracy = as.vector(summary(na.omit(Accuracy))), 
                     Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                     Specificity = as.vector(summary(na.omit(Specificity))),
                     AUC = as.vector(summary(na.omit(AUC))),
                     Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                     Cutoff = as.vector(summary(na.omit(Cutoff))))]

# ---- Results ----------------------------------------------------------------------

# store model diagnostic results
sl.diag[, mod := rep("sl", nrow(sl.diag))]
mods.diag = rbind(mods.diag, sl.diag)

# store the model
sl.list = list("mod" = sl.mod)
mods.list$sl = sl.list

# shutdown the h2o instance if not done already
if("my.dnn" %in% my.library)
{
  h2o.shutdown(prompt = FALSE)
}

# remove objects we no longer need
rm(sl.pred, sl.cv, sl.X, sl.Y, sl.diag, sl.list, sl.mod, my.gbm, my.log,
   my.pen, my.rf, my.dnn, my.svm, my.library, stat, tasks)

rm(output, Xtest, Xtrain, i, Ytest, Ytrain, folds)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Model Predictions ------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Models ----------------------------------------------------------------------

{

# convert mods.diag into long format for plotting purposes
DT = data.table(melt(mods.diag, id.vars = c("stat", "mod")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod, levels = unique(mod))]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value), fill = mod)) +
  geom_bar(stat = "identity", position = "dodge", color = "white") +
  scale_fill_manual(values = mycolors(length(levels(DT$mod)))) +
  labs(x = "Summary Statistic", y = "Value", fill = "Model") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# remove objects we no longer need
rm(diag.plot, DT)

# free memory
gc()

}

# ---- Predictions -----------------------------------------------------------------

{

# ---- logistic regression ---------------------------------------------------------

# build the model
set.seed(42)
mod = glm(Survived ~ ., data = train,
          family = binomial(link = "logit"), 
          control = list(maxit = 100))

# make predictions with the training model using the test set
ynew = predict(mod, test, type = "response")
ynew = as.numeric(ynew >= mods.diag[mod == "log" & stat == "Median", Cutoff])

# build submission
ynew = data.table(PassengerId = (1:nrow(test)) + nrow(train),
                     Survived = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-log.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew)

# ---- penalty regression ----------------------------------------------------------

# extract predictors (X), response (Y), and test set (newX)
X = as.matrix(train[,!"Survived"])
Y = as.numeric(as.character(train$Survived))
newX = as.matrix(test)

# build the model
set.seed(42)
mod = cv.glmnet(x = X, y = Y, family = "binomial", alpha = 0)

# make predictions with the model using the test set
ynew = as.numeric(predict(mod, s = mod$lambda.min, newX, type = "response"))
ynew = as.numeric(ynew >= mods.diag[mod == "pen" & stat == "Median", Cutoff])

# build submission
ynew = data.table(PassengerId = (1:nrow(test)) + nrow(train),
                  Survived = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-pen.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew, X, Y, newX)

# ---- gradient boosting -----------------------------------------------------------

# extract predictors (X), response (Y), and test set (newX)
X = as.matrix(train[, !"Survived"])
Y = as.numeric(as.character(train$Survived))
newX = as.matrix(test)

# build the model
set.seed(42)
mod = xgboost(label = Y, data = X,
              objective = "binary:logistic", eval_metric = "error",
              eta = 0.1, max_depth = 4,
              nrounds = 1000, min_child_weight = 9,
              gamma = 0, verbose = 0, scale.pos.weight = 1.605263,
              subsample = 1, colsample_bytree = 1)

# make predictions with the model using the test set
ynew = as.numeric(predict(mod, newdata = newX))
ynew = as.numeric(ynew >= mods.diag[mod == "gbm" & stat == "Median", Cutoff])

# build submission
ynew = data.table(PassengerId = (1:nrow(test)) + nrow(train),
                  Survived = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-gbm.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew, X, Y, newX)

# ---- random forest ---------------------------------------------------------------

# build the model
set.seed(42)
mod = randomForest(Survived ~ .,
                   data = train,
                   ntree = 1200, 
                   nodesize = 3, 
                   sampsize = c(217, 217), 
                   strata = train$Survived)

# make predictions with the model using the test set
ynew = as.numeric(predict(mod, newdata = test, type = "prob")[,2])
ynew = as.numeric(ynew >= mods.diag[mod == "rf" & stat == "Median", Cutoff])

# build submission
ynew = data.table(PassengerId = (1:nrow(test)) + nrow(train),
                  Survived = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-rf.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew)

# ---- deep neural network ---------------------------------------------------------

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# make train and test into h2o objects
train.h2o = as.h2o(train)
test.h2o = as.h2o(test)

# identify predictors (x) and response (y)
y = "Survived"
x = names(test)

# build the training model
mod = h2o.deeplearning(y = y,
                       x = x,
                       training_frame = train.h2o,
                       hidden = c(115, 96, 77, 58, 39, 20),
                       l1 = 1e-05,
                       l2 = 1e-05,
                       epochs = 10,
                       seed = 42,
                       balance_classes = TRUE,
                       class_sampling_factors = c(1, 1.605263),
                       variable_importances = FALSE)

# make predictions with the training model using the test set
ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = test.h2o))$predict))

# build submission
ynew = data.table(PassengerId = (1:nrow(test)) + nrow(train),
                  Survived = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-dnn.csv", row.names = FALSE)

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# remove objects we no longer need
rm(mod, ynew, train.h2o, test.h2o, x, y)

# ---- support vector machine ------------------------------------------------------

# build the model
set.seed(42)
mod = svm(Survived ~ .,
          data = train,
          cost = 2.201331, 
          gamma = 1/45, 
          class.weights = c("0" = 0.6229508, "1" = 1))

# make predictions with the model using the test set
ynew = as.numeric(as.character(predict(mod, newdata = test)))

# build submission
ynew = data.table(PassengerId = (1:nrow(test)) + nrow(train),
                  Survived = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-svm.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew)

# ---- super learner ---------------------------------------------------------------

# extract predictors (X), response (Y), and test set (newX)
sl.X = as.matrix(train[, !"Survived"])
sl.Y = as.numeric(as.character(train$Survived))
sl.newX = as.matrix(test)

# create Super Learner wrappers

# logistic regression wrapper
my.log = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # build the training model
  set.seed(42)
  mod = glm(y ~ ., data = dat,
            family = binomial(link = "logit"), 
            control = list(maxit = 100))
  
  # make predictions with the training model using the test set
  ynew = predict(mod, data.table(newX), type = "response")
  ynew = as.numeric(ynew >= mods.diag[mod == "log" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# penalty regression wrapper
my.pen = function(Y, X, newX, ...)
{
  # make Y into a factor data type
  Y = factor(Y, levels = 0:1)
  
  # build the training model
  set.seed(42)
  mod = cv.glmnet(x = X, y = Y, family = "binomial", alpha = 0)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, s = mod$lambda.min, newX, type = "response"))
  ynew = as.numeric(ynew >= mods.diag[mod == "pen" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# gradient boosting wrapper
my.gbm = function(Y, X, newX, ...)
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Y, data = X,
                objective = "binary:logistic", eval_metric = "error",
                eta = 0.1, max_depth = 4,
                nrounds = 100, min_child_weight = 9,
                gamma = 0, verbose = 0, scale.pos.weight = 1.605263,
                subsample = 1, colsample_bytree = 1)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = newX))
  ynew = as.numeric(ynew >= mods.diag[mod == "gbm" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# random forest wrapper
my.rf = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # update sampsize
  sampsize = c(217, 217)
  sampsize.update = (nrow(dat) / nrow(sl.X)) * sampsize
  
  # build the training model
  set.seed(42)
  mod = randomForest(y ~ .,
                     data = dat,
                     ntree = 1200, 
                     nodesize = 3, 
                     sampsize = sampsize.update, 
                     strata = dat$y)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = data.table(newX), type = "prob")[,2])
  ynew = as.numeric(ynew >= mods.diag[mod == "rf" & stat == "Median", Cutoff])
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# deep neural network wrapper
my.dnn = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # make dat and newX into h2o objects
  dat.h2o = as.h2o(dat)
  newX.h2o = as.h2o(data.table(newX))
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         hidden = c(115, 96, 77, 58, 39, 20),
                         l1 = 1e-05,
                         l2 = 1e-05,
                         epochs = 10,
                         seed = 42,
                         balance_classes = TRUE,
                         class_sampling_factors = c(1, 1.605263),
                         variable_importances = FALSE)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = newX.h2o))$predict))
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# support vector machine wrapper
my.svm = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := factor(Y, levels = 0:1)]
  
  # build the training model
  set.seed(42)
  mod = svm(y ~ .,
            data = dat,
            cost = 2.201331, 
            gamma = 1/45, 
            class.weights = c("0" = 0.6229508, "1" = 1))
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(predict(mod, newdata = data.table(newX))))
  
  # build a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  
  # free memory
  gc()
  
  return(output)
}

# create a library of the chosen wrappers
my.library = list("my.pen", "my.gbm", "my.rf", "my.dnn", "my.svm")

# initialize the h2o instance if dnn is in sl
if("my.dnn" %in% my.library)
{
  h2o.init()
  h2o.removeAll()
  h2o.no_progress()
}

# build the model
set.seed(42)
mod = SuperLearner(Y = sl.Y, X = sl.X, newX = sl.newX, 
                   family = binomial(), SL.library = my.library, 
                   cvControl = list(stratifyCV = TRUE))

# make predictions with the model using the test set
ynew = as.numeric(mod$SL.predict)
ynew = as.numeric(ynew >= mods.diag[mod == "sl" & stat == "Median", Cutoff])

# build submission
ynew = data.table(PassengerId = (1:nrow(test)) + nrow(train),
                  Survived = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-sl.csv", row.names = FALSE)

# shutdown the h2o instance if dnn is in sl
if("my.dnn" %in% my.library)
{
  h2o.shutdown(prompt = FALSE)
}

# remove objects we no longer need
rm(mod, ynew, sl.X, sl.Y, sl.newX, my.log, my.pen, my.gbm, my.rf, my.dnn, my.svm, my.library)

# free memory
gc()

}

}







