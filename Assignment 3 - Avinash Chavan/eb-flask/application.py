import boto.s3.connection
import time
import datetime
import mysql.connector
import random
import itertools
import urllib
import csv
import memcache
from flask import Flask, render_template, request, redirect, session

access_key = 'AKIAJAVS7AVLUE2BTKKQ'
secret_key = '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'

conn = boto.connect_s3(aws_access_key_id=access_key,aws_secret_access_key=secret_key,)

cnx = mysql.connector.connect(user='sagarchhadia', password='passw0rd',
                              host='awsdbinstance.c0myc1gq6oia.us-east-1.rds.amazonaws.com',
                              database='AWSCacheDB')

mc = memcache.Client(['awscluster.zkvvlb.cfg.use1.cache.amazonaws.com:11211'], debug=1)
mc = 1
print "Connection Done!!"

cursor = cnx.cursor(buffered=True)

application = Flask(__name__)
app = application


@app.route("/")
def Welcome():
    return render_template('index.html')


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file_name = request.files['file_upload'].filename
        print(request.files['file_upload'])
        content = request.files['file_upload'].read()
        bucket_name = getBucket('sagarchhadia11')
        with open(file_name, 'rb') as fileRead:
            plaintext = fileRead.read()
            key = bucket_name.new_key(file_name)
            key.set_contents_from_string(plaintext)
    else:
        print "File limit Reached!!"
        return '<h1>Awesome! File uploaded Successfully.<h1><br><form action="../"><input type="Submit" value="Lets go back"></form>'


@app.route("/select", methods=['GET', 'POST'])
def hello():
    if request.method == "POST":
        #return ("Hello")
        Query = request.form["TextQuery"]
        before = timeBeforeEvent()
        # s = Search1(Query)
        rows = Search(Query)
        after = timeAfterEvent()
        s_time = totalTime(before, after)
        print(s_time)
        s = ""
        s = s_time + '<br>' + s
        for row in rows:
            s = s + '<li>' + row[0] + '<br>' + '</li>'
        return s + '</ol>'
    else:
        return '<H2>Error on Page :(</H2>'


def Search1(Query):
    if isCached(Query.replace(' ', '_')):
        s = 'key is present'
        rows = getKeyValue(Query.replace(' ', '_'))
        for row in rows:
            s = s + '<li>' + row[0] + '<br>' + '</li>'
        return s + '</ol>'

    else:
        s = 'key is not present'
        rows = Search(Query)
        setKeyValue(Query.replace(' ', '_'), rows, 1000)
        for row in rows:
            s = s + '<li>' + row[0] + '<br>' + '</li>'
        return s + '</ol>'


def Search(query):
    cursor.execute(query)
    rows = cursor.fetchall()
    return rows


def insertData(File_URL, File_name):
    loaddata = "LOAD DATA LOCAL INFILE '" + File_URL + File_name + "' INTO TABLE UNPrecip1 FIELDS TERMINATED BY \',\' OPTIONALLY ENCLOSED BY \'\"\' LINES TERMINATED BY \'\n\' IGNORE 1 LINES"
    cursor.execute(loaddata)
    cnx.commit()
    print ("Entered into database")


def isCached(Key_Name):
    if mc.get(Key_Name):
        return 1
    else:
        return 0


def setKeyValue(Key_Name, Value, timeout):
    mc.set(Key_Name, Value, timeout)


def getKeyValue(Key_Name):
    return mc.get(Key_Name)


def createBucket(bucket_name):
    conn.create_bucket(bucket_name)


def getBucket(bucket_name):
    return conn.get_bucket(bucket_name)


def deleteBucket(bucket_name):
    conn.delete_bucket(bucket_name)


def getKey(bucket_name, key_name):
    return bucket_name.get_key(key_name)


def createKey(bucket_name, key_name, local_file):
    new_key = bucket_name.new_key(key_name)
    with open(local_file, 'r') as input:
        CSV_Content_of_File = input.read()
        new_key.set_contents_from_string(CSV_Content_of_File)


def deleteKey(bucket_name, key_name):
    bucket_name.delete_key(key_name)


def downloadingFileAWS(bucket_name, key_name, local_file):
    Key = getKey(bucket_name, key_name)
    Key.get_contents_to_filename(local_file)
    print 'Success ! File downloaded successfully.'


def ListingBucketFiles(bucket_name):
    for key in bucket_name.list():
        print "{name}\t{size}\t{modified}".format(name=key.name, size=key.size, modified=key.last_modified, )


# Printing the name and date of creation of the bucket
def printingBucketInformation():
    for bucket in conn.get_all_buckets():
        print "{name}\t{created}".format(name=bucket.name, created=bucket.creation_date, )


def timeBeforeEvent():
    print 'Time Before Uploading :'
    Time_Before_Upload = time.time()
    print Time_Before_Upload
    return Time_Before_Upload


def timeAfterEvent():
    print 'Time After Uploading :'
    Time_After_Upload = time.time()
    print Time_After_Upload
    return Time_After_Upload


def totalTime(Time_Before_Upload, Time_After_Upload):
    Difference_In_Times = Time_After_Upload - Time_Before_Upload
    Minutes = datetime.datetime.fromtimestamp(Difference_In_Times).minute
    Seconds = datetime.datetime.fromtimestamp(Difference_In_Times).second
    s = 'Time taken to search : {} min {} sec'.format(Minutes, Seconds)
    return s


if __name__ == "__main__":
    app.run(debug=True)