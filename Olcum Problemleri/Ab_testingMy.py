######################################################
# Temel İstatistik Kavramları
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################
# Sampling (Örnekleme)
############################

# Sayı populasyonumuzu olusturuyoruz
populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()

# random seed ayarı örnek cekmek için
np.random.seed(115)

# Örneklem seçmek için
orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()

# Örneklem çekiyor ve seçioruz
np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

# Örneklemlerin ortalamasını alalım
(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


############################
# Descriptive Statistics (Betimsel İstatistikler)
############################

# seabordan tips veriseti cekıoruz
df = sns.load_dataset("tips")
df.describe().T

df.head()

# İlgili değişken için güven aralığı hesabı yapmak
sms.DescrStatsW(df["total_bill"]).tconfint_mean()

# Gelecek bahşişler için bir değerlendirme
sms.DescrStatsW(df["tip"]).tconfint_mean()

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T

# İlgili değişken için eksık degerlerı ucurduktan sonra güven aralığı hesabı yapmak
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

# İlgili değişken için eksık degerlerı ucurduktan sonra güven aralığı hesabı yapmak
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

######################################################
# Correlation (Korelasyon)
######################################################

# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()


df["total_bill"] = df["total_bill"] - df["tip"]

# Saçılım grafiği ile ikisi arasındaki ilişkiyi inceleyelim(pozitif yönlü ,orta sıddetlı bir ilişki var)
df.plot.scatter("tip", "total_bill")
plt.show()

# Matematiksel karşılığı(orta şiddetin biraz üstünde)(iki değişken arasındaki korelasyon gözlemlemesi için kullanırız)
df["tip"].corr(df["total_bill"])

############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İstatistik Ol An Fark var mı?
############################

df = sns.load_dataset("tips")
df.head()

# Ikı grubun ortalamalarına bakalım(sigara içip içmeme durumuna göre)
df.groupby("smoker").agg({"total_bill": "mean"})
# Arada bir fark varmı var gibi gözüküyor ama istastiki olarak bakalım


############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2(
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı:Bir değişkenin dagılımının standart normal dağılımına benzer olup olmadıgının hipotez testidir.
# Varyans Homojenliği:

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

# Normal dagılım kontrolu için sigara içenler için
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Normal dagılım kontrolu için sigara içmeyenler için
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Her ikiside normal dağılıma uymadı.H0 RED ETTI

# Normallik Varsayım sağlanmadıgı için nun parametrık bır test kullanmalıyız

############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

# Varyans Homejenliği için levene testi ile iki farklı gruba göre kontrol yapar.

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Varyans Homojenliği durumuda sağlanmadı bu da rededildi.

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

# Eğer normallik varsayımı sağlanıyorsa kullanılabilen ttest metodu.
# Sadece eger varyans homojenlıgı saglanmıyorsa (equal_var=False) girilir.

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################
# HO rededilemedi.

# nunprametrık ortalama kıyaslama medyan kıyaslama testıdır
test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})
# Fark var gibi gözüküyor bu fark şans eseri ortaya cıkmıs olabilir mi


# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (eşit değildir)  (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark.vardır)
# M1 ve M2=Ana kitle ortalamasının temsilleri.


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  #HO:REDEDİLİR

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #HO:REDEDİLİR

# İki grup içinde varsayım saglanmamaktadır.

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0:Küçük olmadığı için Red edemedik.Bunun anlamı varyanslar homojendir.

# Varsayımlar sağlanmadığı için nonparametrik

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0: Red ed,yorum.Kadın ve erkek yolcuların yas ortalamaları arasında gözlediğimiz
# fark istatistiki olarak da vardır.Fark var ve gorduk!

# 90 280


############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("Olcum Problemleri/datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# Normallik varsayımı sağlanmadığı için nonparametrik.
# Medyan kıyaslaması olarak cıkar yada sıralama kıyaslaması
# H0: reddedilir.

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("Olcum Problemleri/datasets/course_reviews.csv")
df.head()

# İlerlemesi %75 ten fazla olan kursiyerler
df[(df["Progress"] > 75)]["Rating"].mean()

# İlerlemesi %25 ten fazla olan kursiyerler
df[(df["Progress"] < 25)]["Rating"].mean()


test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

# Her iki grubu ayrı arraya koyuyoruz
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

# Kıyaslama metodu ile işlemi yapıyoruz
proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
# H0 red ediyoruz.H1 İKİ ORAN ARASINDA İSTATİKSEL OLARAK ORAN ANLAMNINDA FARKLILIK VARDIR.

basari_sayisi / gozlem_sayilari

############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2 (p1-p2=0)
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

# Bu oranlar için proportions testi için ilk argumanına basarı sayısını  ikinci argumunanına gözlem sayısını ıstıyor.

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Kadın ve erkeklerın hayatta kalma oranları arasında farkı degerlendırdık.
# H0 Reddedilir.Anlamlı bir farklılık yoktur
# H1 Anlamlı bir farklılık vardır.

######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.

# Öyle bir işlem olsun verisetindeki day degıskenını fıltrelemıs olalım ve sonra da shapiro testi yapalım.
# Bir kategorik değişkenin sınıfları üzerinde gezebilecek itaratif bir nesneye çevirdik (Listeye)

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)

# --Hepsi için p değeri 0.05 den kucuk oldugu ıcın H0 red edilir.
# --Hiç birisi için normallik varsayımı saglanmamaktadır:


# H0: Varyans homojenliği varsayımı sağlanmaktadır.
# 4 grup için levene metodu

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 reddedilemez varyans homojenlıgı saglanmaktadır.


# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})

# Medianlarda farklılık var kayda degermı tam bılmıoruz.
# Bu gruplar arasında fark varmı yokmu test etmek ıle grupların ayrı ayrı ortalamaları arası test ayrı bır sey

# HO: Grup ortalamaları arasında istatiksel olarak anlamlı farklılık yoktur

# Varsayım Sağlanıyorsa tek yönlü parametrık test kullanacagız
# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])
# p value degerı 0.05 den kucuk H0 red edilir.
# Dogal olarak gruplar arasında anlamlı bir farklılık vardır.

# Bu varsayımlar saglanmamıstı parametrık yerine nonparamatrik kullanalım
# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

# p value deger 0.05 den kucuk H0 red edilir.
# Görüyoruz ki bu grupların arasında anlamlı bir farklılık vardır.

# Evet fark var ama fark kımden neyden hangı gruptan kaynaklanıyor.

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

# Istatıstıkı olarak ıkılı karsılastırma ıle kayda deger bır farklılık bulunamadı.Ama gruba komple bakınca fark vardı.
# Alfa degerı ıle oyannabılır kıyaslancak deger ıcın ayarlama olabılır.
# Fark yokmus mualamesı yapılabılır.











