embedding = 0.43 # tau
L1mean = LF_model(X_HF_train)
L1mean_up= LF_model(X_HF_train+embedding)
L1mean_dn = LF_model(X_HF_train-embedding)

L2train = torch.hstack((X_HF_train, L1mean, L1mean_up, L1mean_dn)) # Combine the embedded predictions in addition to the LFNN prediction 