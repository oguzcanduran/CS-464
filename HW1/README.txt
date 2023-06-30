- You must have installed Python3, pandas, numpy and matplotlib to run program.

- In order to run the q3main.py, datafiles (x_train.csv,y_train.csv,x_test.csv,y_test.csv) 
must be in the same folder with q3main.py. If the CSV files has different, names you should change them.

- You must change the root variable in the q3main.py to your root variable (where you store q3main.py)
For example for my computer it is:   'C:\Users\Oguz\Desktop\Bilkent\3.2\CS464\hw1'

- In order to run the script use the following command in command prompt:
python path\q3main
For example for my computer the command is: python C:\Users\Oguz\Desktop\Bilkent\3.2\CS464\hw1\q3main.py

-When you called it from command prompt, firstly the program will pring the ratio of the spam mails,
which is the answer of Q2.1.1.Then the program prints 4 different accuracy results, which are Multinomial
Naive Bayes Model,Multinomial Naive Bayes Model with Drichet Prior=5, Multinomial Naive Bayes Model with 
Drichet Prior=10^(-10) and Bernoulli Naive Bayes Model. Also it shows their confusion matrices in a window.

-If you don't want to call the main program, and only want the result of four of those you can use result(i)
function. If i=1, it will show the result of Multinomial Naive Bayes Model. If i=2, it will show the result 
of Multinomial Naive Bayes Model with Drichet Prior=5. If i=3 it will show the result of Multinomial Naive 
Bayes Model with Drichet Prior=10^(-10) and If i=4 it will show the result of Bernoulli Naive Bayes Model.

-In the results parts most of the parameters are default, but if you use one of prediction function you can 
put other datasets. 

