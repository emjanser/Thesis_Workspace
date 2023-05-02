X_LF_train, X_LF_test, Y_LF_train, Y_LF_test = train_test_split(LF_TS, LF(LF_TS), test_size=0.8, shuffle=156)
X_HF_train, X_HF_test, Y_HF_train, Y_HF_test = train_test_split(HF_TS, HF(HF_TS), test_size=0.8, shuffle=156) # Splitting the data sets into training and testing sets per fidelity

# -----Additional pre-processing methods-----
Scaler1 = StandardScaler()
X_train = pandas.DataFrame(Scaler1.fit_transform(X_train)) #  removes the mean and scaling to unit variance

X_train = X_train.fillna(X_train.mean(),inplace=True) # replaces NAN values with mean of the data set