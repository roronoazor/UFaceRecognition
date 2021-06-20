import face_recognition
import os
import json
import time
import numpy


def create_pickled_encodings(path):
    encodings_data = dict()
    if os.path.exists(path) and os.path.isdir(path):
        sub_files = os.listdir(path)
        for s in sub_files:
            if os.path.exists(path + "/" + s) and os.path.isfile(path + "/" + s):
                c = path + "/" + s
                known_image = face_recognition.load_image_file(c)
                image_encoding = face_recognition.face_encodings(known_image)
                if image_encoding:
                    # this check is to avoid out of index error
                    image_encoding = image_encoding[0]

                    # write this encoding to a file to be stored there
                    # if i convert ndarray to list i lose the optimization of the ndarray
                    encodings_data[str(s)] = image_encoding.tolist()
        with open("sample.json", "w") as outfile:
            json.dump(encodings_data, outfile)
#
#
# known_image = face_recognition.load_image_file("known/Adam_Sandler_0002.jpg")
# unknown_image = face_recognition.load_image_file("unknow8.jpeg")
#
# biden_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#
#
# results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
# print(unknown_encoding)
# print(results)


def validate_a_person(path_2_person):
    unknown_image = face_recognition.load_image_file("unknow8.jpeg")

    with open("sample.json") as f:
        encodings_data = json.load(f)

    list_of_encodings = [numpy.asarray(v) for k,v in encodings_data.items()]
    names_of_encodings = [k for k, v in encodings_data.items()]

    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces(list_of_encodings, unknown_encoding)
    names = list()
    count = 0
    for result in results:
        if result == True:
            names.append(names_of_encodings[count])
        count += 1
    print(len(results))
    for name in names:
        print("%s" % (name,))


time_start = time.time()
# create_pickled_encodings("./known")
validate_a_person("unknow8.jpeg")
time_end = time.time()
print("%s", time_end - time_start)