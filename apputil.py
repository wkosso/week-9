import pandas as pd
import numpy as np

# def GroupEstimate(object):
#     def __init__(self, estimate):
#         self.estimate = estimate
    
#     def fit(self, X, y):
#         return None

#     def predict(self, X):
#         return None

class GroupEstimate():
       # Initialize the GroupEstimate with the type of estimate to use ("mean" or "median")
       def __init__(self,estimate):
        if estimate != "mean" and estimate != "median":
            raise ValueError("Unsupported estimate type. Use 'mean' or 'median'.")
        self.estimate = estimate
# Fit the GroupEstimate using the training data X and y
       def fit(self,X,y):
            # Create a series for the target variable y
            series=pd.Series(y, name="y")
            # Combine the features X with the target variable y
            combined= pd.concat([X,series],axis=1)
            self.columns = X.columns
            # Group the combined data by the feature columns and select the target variable
            groupXY = combined.groupby(by=list(X.columns))['y']
            # Calculate the estimate based on the specified method
            if self.estimate == "mean":
                self.estimates = groupXY.mean()
            elif self.estimate == "median":
                self.estimates = groupXY.median()
            else:
                raise ValueError("Unsupported estimate type. Use 'mean' or 'median'.")
            
            return None

       # Predict the target variable for the new data X_
       def predict(self,X_):
           X_ = pd.DataFrame(X_,columns=self.columns)
           # Initialize the list to store predictions and the counter for missing estimates
           predictions=[]
           missing_count=0 
           # Iterate over each row in the new data and retrieve the corresponding estimate
           for index, row in X_.iterrows():
             try:     
                estimate=self.estimates.loc[tuple(row)]
                predictions.append(estimate)
             except KeyError:
                missing_count += 1
                predictions.append(np.nan)

           # Print the number of missing estimates and return the predictions as a NumPy array
           print(f"Missing estimates: {missing_count}")
           return np.array(predictions)