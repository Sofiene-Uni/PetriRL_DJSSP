import pickle

# Replace 'file.pkl' with your pickle file path
with open('d06a.pkl', 'rb') as f:
    data = pickle.load(f)

# Now 'data' contains the object stored in the pickle file
print(len(data))