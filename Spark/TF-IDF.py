from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Initialize Spark
conf = SparkConf().setMaster("local").setAppName("TFIDF")
sc = SparkContext(conf = conf)

# Load documents (one per line).
rawData = sc.textFile("subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names:
documentNames = fields.map(lambda x: x[1])

# hash the words in each document to their term frequencies:
hashingTF = HashingTF(100000)  
tf = hashingTF.transform(documents)

# compute the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)


# the article for "Abraham Lincoln" is in the data, let's search for "Gettysburg" (Lincoln gave a famous speech there):

# First, find out what hash value "Gettysburg" maps to by finding the
# index a sparse vector from HashingTF gives us back:
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# extract the TF*IDF score for Gettsyburg's hash value into
# a new RDD for each document: 
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# zip in the document names:
zippedResults = gettysburgRelevance.zip(documentNames)

# print the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())
