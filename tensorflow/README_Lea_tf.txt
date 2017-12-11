tf.record files müssen selbst erstellt werden, da git Speicher zu klein.
1.) Datei mit Annotationen in cvtennis/annotations suchen.
2.) videoname_test.record und videoname_train.record files in cvtennis/tensorflow/data erstellen
3.) txt python_to_tfrecord_multiclass.py --input_path=../annotations/file_train_test.txt --output_path=/home/lea/Dokumente/FSU/Anwendungspraktikum/tensorflow/data/file_train_test.record --image_folder=/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/file

In /models/training werden checkpoints gespeichert (in gitignore, erst wenn model gut genug für weitere Berechnungen ist)


