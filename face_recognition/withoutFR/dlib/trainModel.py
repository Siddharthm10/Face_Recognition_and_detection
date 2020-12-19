# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

print("[INFO] loading face embeddings...")
data = pickle.loads(open('face_recognition/withoutFR/output/embeddings.pickle', "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print("length ", len(list(set(data["names"]))))

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="poly", degree=4, probability=True)
recognizer.fit(data["embeddings"], labels)
# neigh = KNeighborsClassifier(n_neighbors=len(list(set(data["names"]))))
# neigh.fit(data["embeddings"], labels)
# clf = RandomForestClassifier(n_estimators = 1000, max_depth=2, random_state = 42)
# clf.fit(data["embeddings"], labels)
# write the actual face recognition model to disk
f = open('face_recognition/withoutFR/output/recognizer.pickle', "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open('face_recognition/withoutFR/output/le.pickle', "wb")
f.write(pickle.dumps(le))
f.close()