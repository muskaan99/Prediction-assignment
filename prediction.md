TITLE: Prediction Assignment by Muskaan Parmar

SUMMARY: The goal of this assignment is to use the data of 6
participants obtained from accelerometers on the belt, forearm, arm, and
dumbell and predict the manner in which they exercise. They were asked
to perform barbell lifts correctly and incorrectly in 5 different ways.
Since, people regularly quantify how much of a particular activity they
do, but they rarely quantify how well they do it which would be dealt in
this assignment. We will build a model, use cross validation and thus
make choices. Also, we will use the model on 20 different test cases.

1.  Setting the directory and loading the data

<!-- -->

    setwd("~/R/Coursera 8")
    library(randomForest)

    ## Warning: package 'randomForest' was built under R version 4.0.3

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    library(lattice)
    library(ggplot2)

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

    library(doParallel)

    ## Warning: package 'doParallel' was built under R version 4.0.3

    ## Loading required package: foreach

    ## Warning: package 'foreach' was built under R version 4.0.3

    ## Loading required package: iterators

    ## Warning: package 'iterators' was built under R version 4.0.3

    ## Loading required package: parallel

    library(rpart)

    ## Warning: package 'rpart' was built under R version 4.0.3

    library(caret)   

    ## Warning: package 'caret' was built under R version 4.0.3

    library(rpart.plot)

    ## Warning: package 'rpart.plot' was built under R version 4.0.3

    train <- read.csv("pml-training.csv", na.strings=c("NA","","#DIV/0!"))
    test <- read.csv("pml-testing.csv", na.strings=c("NA", "", "#DIV/0!"))
    #loading data and removing NA,#DIV/0! and blank values from data
    head(train)

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
    ## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
    ## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
    ## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
    ## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
    ##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
    ## 1         no         11      1.41       8.07    -94.4                3
    ## 2         no         11      1.41       8.07    -94.4                3
    ## 3         no         11      1.42       8.07    -94.4                3
    ## 4         no         12      1.48       8.05    -94.4                3
    ## 5         no         12      1.48       8.07    -94.4                3
    ## 6         no         12      1.45       8.06    -94.4                3
    ##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
    ## 1                 NA                  NA                NA                 NA
    ## 2                 NA                  NA                NA                 NA
    ## 3                 NA                  NA                NA                 NA
    ## 4                 NA                  NA                NA                 NA
    ## 5                 NA                  NA                NA                 NA
    ## 6                 NA                  NA                NA                 NA
    ##   skewness_roll_belt.1 skewness_yaw_belt max_roll_belt max_picth_belt
    ## 1                   NA                NA            NA             NA
    ## 2                   NA                NA            NA             NA
    ## 3                   NA                NA            NA             NA
    ## 4                   NA                NA            NA             NA
    ## 5                   NA                NA            NA             NA
    ## 6                   NA                NA            NA             NA
    ##   max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt amplitude_roll_belt
    ## 1           NA            NA             NA           NA                  NA
    ## 2           NA            NA             NA           NA                  NA
    ## 3           NA            NA             NA           NA                  NA
    ## 4           NA            NA             NA           NA                  NA
    ## 5           NA            NA             NA           NA                  NA
    ## 6           NA            NA             NA           NA                  NA
    ##   amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt avg_roll_belt
    ## 1                   NA                 NA                   NA            NA
    ## 2                   NA                 NA                   NA            NA
    ## 3                   NA                 NA                   NA            NA
    ## 4                   NA                 NA                   NA            NA
    ## 5                   NA                 NA                   NA            NA
    ## 6                   NA                 NA                   NA            NA
    ##   stddev_roll_belt var_roll_belt avg_pitch_belt stddev_pitch_belt
    ## 1               NA            NA             NA                NA
    ## 2               NA            NA             NA                NA
    ## 3               NA            NA             NA                NA
    ## 4               NA            NA             NA                NA
    ## 5               NA            NA             NA                NA
    ## 6               NA            NA             NA                NA
    ##   var_pitch_belt avg_yaw_belt stddev_yaw_belt var_yaw_belt gyros_belt_x
    ## 1             NA           NA              NA           NA         0.00
    ## 2             NA           NA              NA           NA         0.02
    ## 3             NA           NA              NA           NA         0.00
    ## 4             NA           NA              NA           NA         0.02
    ## 5             NA           NA              NA           NA         0.02
    ## 6             NA           NA              NA           NA         0.02
    ##   gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
    ## 1         0.00        -0.02          -21            4           22
    ## 2         0.00        -0.02          -22            4           22
    ## 3         0.00        -0.02          -20            5           23
    ## 4         0.00        -0.03          -22            3           21
    ## 5         0.02        -0.02          -21            2           24
    ## 6         0.00        -0.02          -21            4           21
    ##   magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
    ## 1            -3           599          -313     -128      22.5    -161
    ## 2            -7           608          -311     -128      22.5    -161
    ## 3            -2           600          -305     -128      22.5    -161
    ## 4            -6           604          -310     -128      22.1    -161
    ## 5            -6           600          -302     -128      22.1    -161
    ## 6             0           603          -312     -128      22.0    -161
    ##   total_accel_arm var_accel_arm avg_roll_arm stddev_roll_arm var_roll_arm
    ## 1              34            NA           NA              NA           NA
    ## 2              34            NA           NA              NA           NA
    ## 3              34            NA           NA              NA           NA
    ## 4              34            NA           NA              NA           NA
    ## 5              34            NA           NA              NA           NA
    ## 6              34            NA           NA              NA           NA
    ##   avg_pitch_arm stddev_pitch_arm var_pitch_arm avg_yaw_arm stddev_yaw_arm
    ## 1            NA               NA            NA          NA             NA
    ## 2            NA               NA            NA          NA             NA
    ## 3            NA               NA            NA          NA             NA
    ## 4            NA               NA            NA          NA             NA
    ## 5            NA               NA            NA          NA             NA
    ## 6            NA               NA            NA          NA             NA
    ##   var_yaw_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y
    ## 1          NA        0.00        0.00       -0.02        -288         109
    ## 2          NA        0.02       -0.02       -0.02        -290         110
    ## 3          NA        0.02       -0.02       -0.02        -289         110
    ## 4          NA        0.02       -0.03        0.02        -289         111
    ## 5          NA        0.00       -0.03        0.00        -289         111
    ## 6          NA        0.02       -0.03        0.00        -289         111
    ##   accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z kurtosis_roll_arm
    ## 1        -123         -368          337          516                NA
    ## 2        -125         -369          337          513                NA
    ## 3        -126         -368          344          513                NA
    ## 4        -123         -372          344          512                NA
    ## 5        -123         -374          337          506                NA
    ## 6        -122         -369          342          513                NA
    ##   kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm
    ## 1                 NA               NA                NA                 NA
    ## 2                 NA               NA                NA                 NA
    ## 3                 NA               NA                NA                 NA
    ## 4                 NA               NA                NA                 NA
    ## 5                 NA               NA                NA                 NA
    ## 6                 NA               NA                NA                 NA
    ##   skewness_yaw_arm max_roll_arm max_picth_arm max_yaw_arm min_roll_arm
    ## 1               NA           NA            NA          NA           NA
    ## 2               NA           NA            NA          NA           NA
    ## 3               NA           NA            NA          NA           NA
    ## 4               NA           NA            NA          NA           NA
    ## 5               NA           NA            NA          NA           NA
    ## 6               NA           NA            NA          NA           NA
    ##   min_pitch_arm min_yaw_arm amplitude_roll_arm amplitude_pitch_arm
    ## 1            NA          NA                 NA                  NA
    ## 2            NA          NA                 NA                  NA
    ## 3            NA          NA                 NA                  NA
    ## 4            NA          NA                 NA                  NA
    ## 5            NA          NA                 NA                  NA
    ## 6            NA          NA                 NA                  NA
    ##   amplitude_yaw_arm roll_dumbbell pitch_dumbbell yaw_dumbbell
    ## 1                NA      13.05217      -70.49400    -84.87394
    ## 2                NA      13.13074      -70.63751    -84.71065
    ## 3                NA      12.85075      -70.27812    -85.14078
    ## 4                NA      13.43120      -70.39379    -84.87363
    ## 5                NA      13.37872      -70.42856    -84.85306
    ## 6                NA      13.38246      -70.81759    -84.46500
    ##   kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
    ## 1                     NA                      NA                    NA
    ## 2                     NA                      NA                    NA
    ## 3                     NA                      NA                    NA
    ## 4                     NA                      NA                    NA
    ## 5                     NA                      NA                    NA
    ## 6                     NA                      NA                    NA
    ##   skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
    ## 1                     NA                      NA                    NA
    ## 2                     NA                      NA                    NA
    ## 3                     NA                      NA                    NA
    ## 4                     NA                      NA                    NA
    ## 5                     NA                      NA                    NA
    ## 6                     NA                      NA                    NA
    ##   max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
    ## 1                NA                 NA               NA                NA
    ## 2                NA                 NA               NA                NA
    ## 3                NA                 NA               NA                NA
    ## 4                NA                 NA               NA                NA
    ## 5                NA                 NA               NA                NA
    ## 6                NA                 NA               NA                NA
    ##   min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
    ## 1                 NA               NA                      NA
    ## 2                 NA               NA                      NA
    ## 3                 NA               NA                      NA
    ## 4                 NA               NA                      NA
    ## 5                 NA               NA                      NA
    ## 6                 NA               NA                      NA
    ##   amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
    ## 1                       NA                     NA                   37
    ## 2                       NA                     NA                   37
    ## 3                       NA                     NA                   37
    ## 4                       NA                     NA                   37
    ## 5                       NA                     NA                   37
    ## 6                       NA                     NA                   37
    ##   var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
    ## 1                 NA                NA                   NA                NA
    ## 2                 NA                NA                   NA                NA
    ## 3                 NA                NA                   NA                NA
    ## 4                 NA                NA                   NA                NA
    ## 5                 NA                NA                   NA                NA
    ## 6                 NA                NA                   NA                NA
    ##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell
    ## 1                 NA                    NA                 NA               NA
    ## 2                 NA                    NA                 NA               NA
    ## 3                 NA                    NA                 NA               NA
    ## 4                 NA                    NA                 NA               NA
    ## 5                 NA                    NA                 NA               NA
    ## 6                 NA                    NA                 NA               NA
    ##   stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x gyros_dumbbell_y
    ## 1                  NA               NA                0            -0.02
    ## 2                  NA               NA                0            -0.02
    ## 3                  NA               NA                0            -0.02
    ## 4                  NA               NA                0            -0.02
    ## 5                  NA               NA                0            -0.02
    ## 6                  NA               NA                0            -0.02
    ##   gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
    ## 1             0.00             -234               47             -271
    ## 2             0.00             -233               47             -269
    ## 3             0.00             -232               46             -270
    ## 4            -0.02             -232               48             -269
    ## 5             0.00             -233               48             -270
    ## 6             0.00             -234               48             -269
    ##   magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
    ## 1              -559               293               -65         28.4
    ## 2              -555               296               -64         28.3
    ## 3              -561               298               -63         28.3
    ## 4              -552               303               -60         28.1
    ## 5              -554               292               -68         28.0
    ## 6              -558               294               -66         27.9
    ##   pitch_forearm yaw_forearm kurtosis_roll_forearm kurtosis_picth_forearm
    ## 1         -63.9        -153                    NA                     NA
    ## 2         -63.9        -153                    NA                     NA
    ## 3         -63.9        -152                    NA                     NA
    ## 4         -63.9        -152                    NA                     NA
    ## 5         -63.9        -152                    NA                     NA
    ## 6         -63.9        -152                    NA                     NA
    ##   kurtosis_yaw_forearm skewness_roll_forearm skewness_pitch_forearm
    ## 1                   NA                    NA                     NA
    ## 2                   NA                    NA                     NA
    ## 3                   NA                    NA                     NA
    ## 4                   NA                    NA                     NA
    ## 5                   NA                    NA                     NA
    ## 6                   NA                    NA                     NA
    ##   skewness_yaw_forearm max_roll_forearm max_picth_forearm max_yaw_forearm
    ## 1                   NA               NA                NA              NA
    ## 2                   NA               NA                NA              NA
    ## 3                   NA               NA                NA              NA
    ## 4                   NA               NA                NA              NA
    ## 5                   NA               NA                NA              NA
    ## 6                   NA               NA                NA              NA
    ##   min_roll_forearm min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
    ## 1               NA                NA              NA                     NA
    ## 2               NA                NA              NA                     NA
    ## 3               NA                NA              NA                     NA
    ## 4               NA                NA              NA                     NA
    ## 5               NA                NA              NA                     NA
    ## 6               NA                NA              NA                     NA
    ##   amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
    ## 1                      NA                    NA                  36
    ## 2                      NA                    NA                  36
    ## 3                      NA                    NA                  36
    ## 4                      NA                    NA                  36
    ## 5                      NA                    NA                  36
    ## 6                      NA                    NA                  36
    ##   var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
    ## 1                NA               NA                  NA               NA
    ## 2                NA               NA                  NA               NA
    ## 3                NA               NA                  NA               NA
    ## 4                NA               NA                  NA               NA
    ## 5                NA               NA                  NA               NA
    ## 6                NA               NA                  NA               NA
    ##   avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
    ## 1                NA                   NA                NA              NA
    ## 2                NA                   NA                NA              NA
    ## 3                NA                   NA                NA              NA
    ## 4                NA                   NA                NA              NA
    ## 5                NA                   NA                NA              NA
    ## 6                NA                   NA                NA              NA
    ##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
    ## 1                 NA              NA            0.03            0.00
    ## 2                 NA              NA            0.02            0.00
    ## 3                 NA              NA            0.03           -0.02
    ## 4                 NA              NA            0.02           -0.02
    ## 5                 NA              NA            0.02            0.00
    ## 6                 NA              NA            0.02           -0.02
    ##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
    ## 1           -0.02             192             203            -215
    ## 2           -0.02             192             203            -216
    ## 3            0.00             196             204            -213
    ## 4            0.00             189             206            -214
    ## 5           -0.02             189             206            -214
    ## 6           -0.03             193             203            -215
    ##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
    ## 1              -17              654              476      A
    ## 2              -18              661              473      A
    ## 3              -18              658              469      A
    ## 4              -16              658              469      A
    ## 5              -17              655              473      A
    ## 6               -9              660              478      A

    print("After removing NA,#DIV/0! and blank values from the two datasets:")

    ## [1] "After removing NA,#DIV/0! and blank values from the two datasets:"

    print("Dimension of training data=")

    ## [1] "Dimension of training data="

    dim(train)

    ## [1] 19622   160

    print("Dimension of testing data=")

    ## [1] "Dimension of testing data="

    dim(test)

    ## [1]  20 160

2.Data pre-processing

    train2<-train[,-c(1:7)]
    test2 <-test[,-c(1:7)]
    #removing index, timestamp, new window, num window and subject name i.e. first 7 columns
    print("After removing non predictors from the two datasets:")

    ## [1] "After removing non predictors from the two datasets:"

    print("Dimension of training data=")

    ## [1] "Dimension of training data="

    dim(train2)

    ## [1] 19622   153

    print("Dimension of testing data=")

    ## [1] "Dimension of testing data="

    dim(test2)

    ## [1]  20 153

    #checking for non zero values in training dataset
    train3<-nzv(train2[,-ncol(train2)],saveMetrics=TRUE)
    row(train3)

    ##        [,1] [,2] [,3] [,4]
    ##   [1,]    1    1    1    1
    ##   [2,]    2    2    2    2
    ##   [3,]    3    3    3    3
    ##   [4,]    4    4    4    4
    ##   [5,]    5    5    5    5
    ##   [6,]    6    6    6    6
    ##   [7,]    7    7    7    7
    ##   [8,]    8    8    8    8
    ##   [9,]    9    9    9    9
    ##  [10,]   10   10   10   10
    ##  [11,]   11   11   11   11
    ##  [12,]   12   12   12   12
    ##  [13,]   13   13   13   13
    ##  [14,]   14   14   14   14
    ##  [15,]   15   15   15   15
    ##  [16,]   16   16   16   16
    ##  [17,]   17   17   17   17
    ##  [18,]   18   18   18   18
    ##  [19,]   19   19   19   19
    ##  [20,]   20   20   20   20
    ##  [21,]   21   21   21   21
    ##  [22,]   22   22   22   22
    ##  [23,]   23   23   23   23
    ##  [24,]   24   24   24   24
    ##  [25,]   25   25   25   25
    ##  [26,]   26   26   26   26
    ##  [27,]   27   27   27   27
    ##  [28,]   28   28   28   28
    ##  [29,]   29   29   29   29
    ##  [30,]   30   30   30   30
    ##  [31,]   31   31   31   31
    ##  [32,]   32   32   32   32
    ##  [33,]   33   33   33   33
    ##  [34,]   34   34   34   34
    ##  [35,]   35   35   35   35
    ##  [36,]   36   36   36   36
    ##  [37,]   37   37   37   37
    ##  [38,]   38   38   38   38
    ##  [39,]   39   39   39   39
    ##  [40,]   40   40   40   40
    ##  [41,]   41   41   41   41
    ##  [42,]   42   42   42   42
    ##  [43,]   43   43   43   43
    ##  [44,]   44   44   44   44
    ##  [45,]   45   45   45   45
    ##  [46,]   46   46   46   46
    ##  [47,]   47   47   47   47
    ##  [48,]   48   48   48   48
    ##  [49,]   49   49   49   49
    ##  [50,]   50   50   50   50
    ##  [51,]   51   51   51   51
    ##  [52,]   52   52   52   52
    ##  [53,]   53   53   53   53
    ##  [54,]   54   54   54   54
    ##  [55,]   55   55   55   55
    ##  [56,]   56   56   56   56
    ##  [57,]   57   57   57   57
    ##  [58,]   58   58   58   58
    ##  [59,]   59   59   59   59
    ##  [60,]   60   60   60   60
    ##  [61,]   61   61   61   61
    ##  [62,]   62   62   62   62
    ##  [63,]   63   63   63   63
    ##  [64,]   64   64   64   64
    ##  [65,]   65   65   65   65
    ##  [66,]   66   66   66   66
    ##  [67,]   67   67   67   67
    ##  [68,]   68   68   68   68
    ##  [69,]   69   69   69   69
    ##  [70,]   70   70   70   70
    ##  [71,]   71   71   71   71
    ##  [72,]   72   72   72   72
    ##  [73,]   73   73   73   73
    ##  [74,]   74   74   74   74
    ##  [75,]   75   75   75   75
    ##  [76,]   76   76   76   76
    ##  [77,]   77   77   77   77
    ##  [78,]   78   78   78   78
    ##  [79,]   79   79   79   79
    ##  [80,]   80   80   80   80
    ##  [81,]   81   81   81   81
    ##  [82,]   82   82   82   82
    ##  [83,]   83   83   83   83
    ##  [84,]   84   84   84   84
    ##  [85,]   85   85   85   85
    ##  [86,]   86   86   86   86
    ##  [87,]   87   87   87   87
    ##  [88,]   88   88   88   88
    ##  [89,]   89   89   89   89
    ##  [90,]   90   90   90   90
    ##  [91,]   91   91   91   91
    ##  [92,]   92   92   92   92
    ##  [93,]   93   93   93   93
    ##  [94,]   94   94   94   94
    ##  [95,]   95   95   95   95
    ##  [96,]   96   96   96   96
    ##  [97,]   97   97   97   97
    ##  [98,]   98   98   98   98
    ##  [99,]   99   99   99   99
    ## [100,]  100  100  100  100
    ## [101,]  101  101  101  101
    ## [102,]  102  102  102  102
    ## [103,]  103  103  103  103
    ## [104,]  104  104  104  104
    ## [105,]  105  105  105  105
    ## [106,]  106  106  106  106
    ## [107,]  107  107  107  107
    ## [108,]  108  108  108  108
    ## [109,]  109  109  109  109
    ## [110,]  110  110  110  110
    ## [111,]  111  111  111  111
    ## [112,]  112  112  112  112
    ## [113,]  113  113  113  113
    ## [114,]  114  114  114  114
    ## [115,]  115  115  115  115
    ## [116,]  116  116  116  116
    ## [117,]  117  117  117  117
    ## [118,]  118  118  118  118
    ## [119,]  119  119  119  119
    ## [120,]  120  120  120  120
    ## [121,]  121  121  121  121
    ## [122,]  122  122  122  122
    ## [123,]  123  123  123  123
    ## [124,]  124  124  124  124
    ## [125,]  125  125  125  125
    ## [126,]  126  126  126  126
    ## [127,]  127  127  127  127
    ## [128,]  128  128  128  128
    ## [129,]  129  129  129  129
    ## [130,]  130  130  130  130
    ## [131,]  131  131  131  131
    ## [132,]  132  132  132  132
    ## [133,]  133  133  133  133
    ## [134,]  134  134  134  134
    ## [135,]  135  135  135  135
    ## [136,]  136  136  136  136
    ## [137,]  137  137  137  137
    ## [138,]  138  138  138  138
    ## [139,]  139  139  139  139
    ## [140,]  140  140  140  140
    ## [141,]  141  141  141  141
    ## [142,]  142  142  142  142
    ## [143,]  143  143  143  143
    ## [144,]  144  144  144  144
    ## [145,]  145  145  145  145
    ## [146,]  146  146  146  146
    ## [147,]  147  147  147  147
    ## [148,]  148  148  148  148
    ## [149,]  149  149  149  149
    ## [150,]  150  150  150  150
    ## [151,]  151  151  151  151
    ## [152,]  152  152  152  152

3.Partitioning train data into validation(testing) set and training set

    intr<- createDataPartition(train2$classe, p = 0.6, list = FALSE)
    training<- train2[intr,]#training set(60%)
    validation<- train2[-intr,]#validation set(40%)
    print("After partitioning training data into validation set(40%) and training set(60%) :")

    ## [1] "After partitioning training data into validation set(40%) and training set(60%) :"

    print("Dimension of training set=")

    ## [1] "Dimension of training set="

    dim(training)

    ## [1] 11776   153

    print("Dimension of validation set=")

    ## [1] "Dimension of validation set="

    dim(validation)

    ## [1] 7846  153

4.Model building by cross validation using Random Forest algorithm

    mfn <- "myModel.RData"
    if (!file.exists(mfn)) 
    {
      nc <- makeCluster(detectCores() - 1)
      registerDoParallel(cores=nc)
      getDoParWorkers() # 3    
      
      myModel  <- train(classe ~ ., data = training, method = "rf", metric = "Accuracy",         preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4, p= 0.60, allowParallel = TRUE ) 
      )
      save(myModel , file = "myModel.RData")
      stopCluster(nc)
    }else 
    {
      load(file = mfn, verbose = TRUE)
    }

    ## Loading objects:
    ##   myModel

    print(myModel, digits=4)

    ## Random Forest 
    ## 
    ## 11776 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## Pre-processing: centered (52), scaled (52) 
    ## Resampling: Cross-Validated (4 fold) 
    ## Summary of sample sizes: 8833, 8831, 8832, 8832 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy  Kappa 
    ##    2    0.9881    0.9850
    ##   27    0.9875    0.9842
    ##   52    0.9783    0.9726
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

5.Prediction

    predTest <- predict(myModel, newdata=validation)

6.Confusion Matrix

    confusionMatrix(predTest, factor(validation$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2230    2    0    0    0
    ##          B    2 1512   11    0    0
    ##          C    0    4 1356   14    0
    ##          D    0    0    1 1272    2
    ##          E    0    0    0    0 1440
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9954          
    ##                  95% CI : (0.9937, 0.9968)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9942          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9991   0.9960   0.9912   0.9891   0.9986
    ## Specificity            0.9996   0.9979   0.9972   0.9995   1.0000
    ## Pos Pred Value         0.9991   0.9915   0.9869   0.9976   1.0000
    ## Neg Pred Value         0.9996   0.9991   0.9981   0.9979   0.9997
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2842   0.1927   0.1728   0.1621   0.1835
    ## Detection Prevalence   0.2845   0.1944   0.1751   0.1625   0.1835
    ## Balanced Accuracy      0.9994   0.9970   0.9942   0.9943   0.9993

The out of sample error is 0.0037. The accuracy is 0.9964 and lies
within the 95% confidence interval.

7.Complete data about the model

    myModel$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.78%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3345    3    0    0    0 0.0008960573
    ## B   16 2255    8    0    0 0.0105309346
    ## C    0   16 2037    1    0 0.0082765336
    ## D    0    0   42 1886    2 0.0227979275
    ## E    0    0    0    4 2161 0.0018475751

    varImp(myModel)

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                   Overall
    ## roll_belt          100.00
    ## yaw_belt            77.09
    ## magnet_dumbbell_z   69.90
    ## pitch_forearm       64.13
    ## magnet_dumbbell_y   63.00
    ## pitch_belt          57.83
    ## magnet_dumbbell_x   53.86
    ## roll_forearm        46.29
    ## accel_dumbbell_y    44.05
    ## accel_belt_z        42.33
    ## magnet_belt_z       42.26
    ## roll_dumbbell       41.21
    ## magnet_belt_y       39.30
    ## accel_dumbbell_z    36.27
    ## roll_arm            32.35
    ## accel_forearm_x     32.26
    ## gyros_belt_z        31.35
    ## accel_dumbbell_x    28.59
    ## yaw_dumbbell        28.48
    ## accel_arm_x         27.76

1.  Quiz Coursera The testing is now performed on the Quiz set.

<!-- -->

    print(predict(myModel, newdata=test2))

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

The above sequence is the set of answers obtained for the Quiz.

CITATIONS: The data for this project comes from :
<a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" class="uri">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a>
