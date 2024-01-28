# WdsiSemProject
## Opis wybranej koncepcji projektu
  Do projektu rozpoznawania marek i modeli pojazdów wykorzystaliśmy metodę z użyciem głębokich sieci neuronowych, które wykorzystywały techniki rozpoznawania obrazu, tak by rozróżniać konkretne elementy nadwozia samochodu, które charakteryzują daną markę lub model, by na wyjściu model zwrócił dokładną markę oraz model pojazdu. Do naszego projektu użyliśmy modeli sieci takich jak VGG16 oraz api Keras które umożliwia szkolenie sieci neuronowych. Do normalizacji zdjęć z naszej bazy użyliśmy biblioteki tensorflow, która jest bardzo często używana w uczeniu maszynowym oraz w głębokich sieciach neuronowych. Użyliśmy metody sieci ponieważ ma charakteryzuje się ona wysoką dokładnością w rozpoznawaniu obrazów, oraz stosunkowo dobrze radzi sobie przy wyciąganiu kluczowych dla funkcjonowania modelu elementów ze zdjęcia pomimo potrzeby posiadania dużej bazy do treningu. Podczas tworzenia naszego projektu posłużyliśmy się zbiorem danych pobranych ze strony kaggle.com, które posiadają bazę danych samochodów dobranych przez uniwersytet stanforda(link do bazy jest umieszczony w bibliografii). Głębokie sieci neuronowe przetwarzają informacje w sposób ludzki mózg, istnieje kilka lub wiele warstw które na każdej z nich dokonują wielu porównań czy też obliczeń potrzebnych do przetworzenia (w naszym przypadku) obrazu, każda z nich posiada pewną wagę a co za tym idzie jest w stanie powiedzieć jak bardzo dany sygnał jest istotny. By móc zrealizować model rozpoznawania modeli pojazdów potrzebne są zdjęcia przykładowo z kamer monitoringu, zdjęć wykonanych przez fotoradary czy też jakichkolwiek osób które wykonały zdjęcie danemu pojazdowi. Wyjściem naszego algorytmu będzie odpowiedź, jaki model jest przypisany do zdjęcia danego algorytmowi na wejściu. <br/>
  Spośród dostępnych metod uznaliśmy ją za najlepszą, ponieważ w przypadku Support Vector Machines zbyt duża ilość danych, która była na wejściu jest w postaci obrazów co przełożyłoby się na bardzo niską skuteczność dla tak dużej bazy danych, skuteczniejsze okazałaby się ta metoda gdybyśmy posiadali mniejszą ilość danych, lub należałoby spośród nich dokonywać porównania pewnych elementów pojazdu a nie to był cel stworzenia naszego modelu.
Zrezygnowaliśmy również z wykorzystania metody przetwarzania obrazów, które w istocie jest efektywne, natomiast wymaga ręcznego zaprojektowania cech, które wykorzystałby algorytm, co wymaga poświęcenia dużej ilości czasu, kiedy mamy do czynienia z obrazami o wyższym stopniu złożoności. <br/>
  Użyta baza posiadała ponad 16 000 zdjęć, które zostały sklasyfikowane w 196 różnych modelach samochodów. Potrzebne były dane csv, w której zapisane były informacje na temat zdjęcia, oraz sklasyfikowanie go w konkretnej grupie modelu. Są to dane dostępne publicznie, które można pobrać ze strony Kaggle.com. 

## Uruchomienie sieci i sprawdzenie jej działania na przykaładowych zdjęciach
Aby uruchomić model należy użyc poniższej komendy w celu uruchomienia testu
```
[cmd] python working_model.py
```

## Bibliografia
- “Everything you need to know about VGG16” by Rohini G <br/>
https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918
- “Introduction to machine learning”  by Weights & Biases <br/>
https://youtube.com/playlist?list=PLD80i8An1OEHSai9cf-Ip-QReOVW76PlB&si=yoLy6-5InhrUMJ4Rhttps://ekursy.put.poznan.pl/course/view.php?id=34988
- “Deep Neural Networks Tutorial with TensorFlow” <br/>
https://medium.com/analytics-vidhya/deep-neural-networks-tutorial-with-tensorflow-d545e4417977
-“Detekcja znaków drogowych na podstawie zdjęć” by Politechnika Wrocławska i Politechnika Łódzka
https://wyd.edu.pl/images/Monografie/Wyzwania_gospodarcze_2018/Wyzwania_2018_Bartczak_Kacprowicz.pdf
- “Tensorflow documentation” <br/>
https://www.tensorflow.org/guide?hl=pl
- “Keras documentation” <br/>
https://keras.io/api/
- “Deep learning with Tensorflow, Keras and Python” <br/>
https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO
- “Stanford Cars Dataset” <br/>
https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

