# Setting the complxity of the synthetic functions 
def LF(y): # LF output function  
    return 0.5 * (6 * y - 2) ** 2 * np.sin(12 * y - 4) + 10 * (y - 0.5) - 5

def HF(y): # HF output function
    return (6 * y - 2)**2 * np.sin(12 * y - 4)