# Kompresija autoenkoderom

## Ideja
Kompresija podataka s gubicima (eng. lossy compression) temelji se na opažanju
da kvaliteta ulaznog skupa podataka (koji se želi kompresirati) ne ovisi jednako
o svakom pojedinom podatku unutar skupa, već neki od njih kvaliteti doprinose više,
a drugi manje. Stoga nije jednako važno zadržati sve podatke, već je neke moguće i
odbaciti ili znatno umanjiti, bez da utjecaj na kvalitetu reprodukcije bude značajan.
Taj spomenuti princip koristi i klasični JPEG algoritam za kompresiju slika.

Ista je ideja primjenjiva i na pokušaj kompresije slika korištenjem neuronskih mreža.
Upravo je ekstrakcija važnih značajki jedna od zadaća u kojima su neuronske mreže
posebno dobre, a među njima konkretno po tom pitanju najviše obećavaju autoenkoderi.

Autoenkoderi su vrsta neuronskih mreža kojima je idejna svrha nenadzirano naučiti
najučinkovitiji prikaz danih im podataka, s učestalom primjenom u redukciji
dimenzionalnosti. Upravo iz tog razloga autoenkoderi su posebno prikladni i za
kompresiju podataka, jer se njihova sposobnost manjedimenzionalnog prikaza ulaznih
podataka može iskoristiti za sažimanje i uklanjanje nepotrebnih informacija, i
zadržavanje samo onih koje su bitne za kvalitetu rezultata i mogućnost rekonstukcije
izvornika.

Konkretno, za primjenu u kompresiji slika, autoenkoderi se mogu koristiti na način
da se treniraju nad skupom slika koje se želi kompresirati vođeni ciljem da se
rekonstrukcija ulaza na krajnjem izlazu autoenkodera što bolje podudara s originalom.
Jednom kada se treniranjem dosegne zadovoljavajuća razina sličnosti izlaza s ulazom,
slike se tim autoenkoderom mogu kompresirati tako da se autoenkoder razdijeli na
dva dijela - prvi, koji od ulaza u jednom ili više koraka stvara njegovu sažetu
skrivenu reprezentaciju, i drugi, koji iz te skrivene reprezentacije ponovno
rekonstruira ulaz. Tada taj prvi dio predstavlja enkoder slike, rezultat čije je
primjene upravo ta podatkovno sažeta reprezentacija, a drugi dio autoenkodera
predstavlja dekoder koji vrši rekonstrukciju originala iz kompresirane slike.

Sama arhitektura autoenkodera može se prilagođavati složenosti slika koje se želi
kompresirati, pa tako za jednostavnije i dimenzijama manje slike dostatan može biti
i jednoslojni autoenkoder, dok se za slike većih dimenzija i složenosti mogu koristiti
i višeslojni ili duboki autoenkoderi.

Mjera kojom se najčešće vrši mjerenje kvalitete algoritama kompresije, tj. njihove
sposobnosti zadržavanja relevantnih informacija nakon rekonstrukcije originala,
jest srednja kvadratna pogreška. Ona se računa prema izrazu
$MSE = \frac{1}{MN}\sum_{x=1}^M\sum_{y=1}^N{(I(x,y) - I'(x,y))^2}$,
i iskazuje koliko se u prosjeku razlikuju vrijednosti pojedinih piksela dvaju slika,
uzimajući pritom u većoj mjeri ona mjesta gdje su razlike značajnije (zbog
kvadrata razlike). I upravo se navedena mjera srednje kvadratne pogreške može
iskoristiti i kao funkcija pogreške pri samom treniranju autoenkodera.

## Rezultati

Opisana ideja provedena je u djelo (GitHub: https://github.com/zjurelinac/AEC)
kroz dva eksperimenta:

1. U prvom eksperimentu različiti autoenkoderi trenirani su na MNIST skupu podataka,
koji se sastoji od 60.000 slika dimenzija 28\*28 koje prikazuju različite rukom
pisane znamenke od 0 do 9. Iskušani su jednoslojni autoenkoderi različitih veličina
skrivenog sloja (32, 48 i 64), kao i duboki autoenkoder sa slojevima 128, 64 i 32.
Ostvareni rezultati prikazani su u tablici.

| Autoenkoder | Dimenzije | Epohe | Konačni MSE | Veličina reprezentacije |
+-------------+-----------+-------+-------------+-------------------------+
| 1-slojni    | 32        | 100   | 0.009392    | 32*2B                   |
| 1-slojni    | 48        | 100   | 0.005413    | 48*2B                   |
| 1-slojni    | 64        | 75*   | 0.003568    | 64*2B                   |
| Višeslojni  | 128,64,32 | 100   | 0.006001    | 32*2B                   |
| Višeslojni  | 256,64,24 | 100   | 0.007231    | 24*2B                   |

Slike rezultata (uvećane, originali su 28\*28):
[sa32m.png]
[sa48m.png]
[sa64m.png]
[da128m.png]
[da256m.png]


2. U drugom eksperimentu različiti autoenkoderi trenirani su na CIFAR10 skupu
podataka, točnije na podskupu slika koje pripadaju istoj kategoriji (odabranoj
kategoriji 2 = pticama). CIFAR10 skup se sastoji od 60.000 slika u boji dimenzija
32\*32\*3, a prikazuje različite objekte koji pripadaju jednoj od 10 kategorija.
Jednoslojni autoenkoderi su se u ovom zadatku pokazali nedostatnima jer nisu mogli
na adekvatan način opisati znatno veću kompleksnost ovog skupa podataka. Stoga je
više pažnje povjereno višeslojnim autoenkoderima, koji su pokazali bolje, iako ne
i idealne rezultate.

| Autoenkoder | Dimenzije  | Epohe | Konačni MSE | Veličina reprezentacije |
+-------------+------------+-------+-------------+-------------------------+
| 1-slojni    | 256        | 100   | 0.012762     | 256*2B                 |
| Višeslojni  | 256,192,64 | 50    | 0.010943     | 64*2B                  |
| Višeslojni  | 512,128,64 | 100   | 0.009328     | 64*2B                  |

Slike rezultata (uvećane, originali su 32\*32):
[sa256c2.png]
[da256c2.png]
[da512c2.png]


Treniranje autoenkodera na računalima bez GPU jedinica i snažnijih procesora
pokazalo se vrlo dugotrajnim, u rasponu od 10 do 50 sekundi po epohi
(ovisno o vrsti autoenkodera i broju parametara, te veličini skupa za treniranje).

Također, u veličinu kompresirane slike nije uračunat i zapis težina samog autoenkodera,
koji, ovisno o arhitekturi autoenkodera, može imati između 200kB i 20MB, što je
svakako nezanemariv broj. No s druge strane, razine kompresije pojedinačnih slika
su, čak i uz ovako rudimentarnu obradu (bez prilagodbe ulaza, odvojenog razmatranja
kanala boja, pretvorbi u druge prostore boja, binarne kompresije rezultantnih
težina i sl.), bile izrazito dobre, za MNIST skup podataka do i više od 15 puta.
Stoga, kada bi se uzeo u obzir cjelokupni MNIST skup podataka koji sadrži 60.000
slika, i on kompresirao na opisani način, utjecaj zapisa težina autoenkodera
značajno bi se umanjio, pa čak i postao nezamjetan. Ukupni kompresijski faktor bi
tada bio (npr. za 1-slojni autoenkoder dimenzije 32):

$$c_{AE} = \frac{S}{S_c + W_{AE}}= \frac{M\cdot N}{D\cdot N + W_{AE}} = \frac{M}{D+\frac{W_{AE}}{N}} = 11.3$$
(S_c = veličina kompresiranog skupa, S = veličina originalnog skupa, M = veličina jedne slike = 784B,
D = veličina skrivene reprezentacije autoenkodera = 32\*2B, W_AE = veličina zapisa težina = 309kB,
N = broj slika u skupu = 60.000)

što je zaista značajan faktor. Za usporedbu, JPEG algoritam na slikama ovih dimenzija
ostvaruje kompresijski faktor od samo 1.05 (784B original, 749B JPEG, jer se u svaku
pojedinačnu sliku pohranjuje i odgovarajuće zaglavlje), što ga za pohranu cjelokupnog
MNIST skupa čini barem 10 puta lošijim od pristupa s autoenkoderom.

## Zaključak

Ovim je radom uspješno iskušano kompresiranje slika pomoću neuronskih mreža, tj.
jedne njihove vrste - autoenkodera. Sa strane kompresijskog faktora, ostvareni su
obećavajući rezultati (faktori 10 i više), uz zadržavanje dobrih svojstava
reprodukcije originala i izbjegavanje ikakvih oku neugodnih artefakata
(kao što su zrnatost i pikselizacija). Vremenski, treniranje autoenkodera na
računalima slabije snage i bez GPU jedinica bilo je iznimno dugotrajno, ali kada
su jednom istrenirani, trajanje kompresije slika bilo je nezamjetno. Također,
potencijalni problem autoenkodera jest i potreba da se za dekodiranje zapamte i
njihove interne težine, što ih čini neprikladnima za kompresiju pojedinačnih slika.
No pri kompresiji većeg skupa slikovnih podataka, autoenkoderi su pokazali izvrsne
kompresijske performanse, te se zasad čini da bi upravo to moglo biti područje u
kojemu su oni namoćni nad svim drugim metodama.

Kompresija slika autoenkoderima svakako izgleda obećavajuće, i uz daljnji rad na
tom području, kako s teoretske, tako i tehničke strane, mogao bi dovesti do daljnjih
značajnih napredaka. Primjerice, u ovome radu uopće nije bila iskušana kompresija
dubokim konvolucijskim autoenkoderima, koji imaju potencijal kompresiju učiniti
višestruko boljom te primjenjivom i na slike daleko većih dimenzija. Također, rad
na tehničkim aspektima algoritma, poput predobrade ulaza, razdvajanja, konverzije
i zasebne obrade kanala slike, modifikacije operacija autoenkodera da budu prikladnije
za cjelobrojne vrijednosti ulaza, te kompresije kodirane reprezentacije slike isto
tako može ovaj postupak učiniti još mnogo uspješnijim i primjenjivijim.
