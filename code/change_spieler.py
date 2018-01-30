
file = '/home/mueller/code/python/Anwendungspraktikum/cvtennis/annotations/GoProAnnotationSpieler_test.txt'
fout = '/home/mueller/code/python/Anwendungspraktikum/cvtennis/annotations/GoProAnnotationSpieler12_test.txt'

countS1 = 0
countS2 = 0

with open(fout, 'w') as fout:
	with open(file, 'r') as f:
		for row in f:
			imgname_, classes_, Xs_, Ys_, Xe_, Ye_ = row.split()

			if int(Xs_) < 1000 and int(Ys_) < 1000 and int(Xe_) < 1000 and int(Ye_) < 1000:
				classes_ = 'Spieler2'
				countS1 += 1
			else:
				classes_ = 'Spieler1'
				countS2 += 1
			wr = "{} {} {} {} {} {}\n".format(imgname_, classes_, Xs_, Ys_, Xe_, Ye_)
			fout.write(wr)

print(countS1, countS2)