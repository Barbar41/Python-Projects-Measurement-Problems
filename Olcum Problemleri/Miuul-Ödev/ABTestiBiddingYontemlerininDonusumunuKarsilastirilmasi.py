######
# AB Testi ile Bidding Yöntemlerinin Dönüşümünü Karşılaştırılması
######

##############################################
# İŞ PROBLEMİ
##############################################
# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif olarak yeni bir teklif türü olan "average bidding"’i tanıttı.
# Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test etmeye karar verdi ve average bidding'in maximumbidding'den /
# Daha fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.
# A/B testi 1 aydır devam ediyor ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için nihai başarı ölçütü Purchase'dır.
# Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.

################################################
# Veri Seti Hikayesi
################################################
# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.
# Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır.
# Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# 4 Değişken   40 Gözlem     26 KB
# Impression : Reklam görüntüleme sayısı
# Click      : Görüntülenen reklama tıklama sayısı
# Purchase   : Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning    : Satın alınan ürünler sonrası elde edilen kazanç

##########################
# Proje Görevleri
#########################

##################################
# Görev 1: Veriyi Hazırlama ve Analiz Etme
#################################
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


# Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

dataframe_control=pd.read_excel("Olcum Problemleri/datasets/ab_testing.xlsx", sheet_name="Control Group")
dataframe_test=pd.read_excel("Olcum Problemleri/datasets/ab_testing.xlsx", sheet_name="Test Group")

df_control= dataframe_control.copy()
df_test=dataframe_test.copy()

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

def check_df(dataframe, head=5):
    print("############################ Shape ###############")
    print(dataframe.shape)
    print("############################ Types ###############")
    print(dataframe.dtypes)
    print("############################ Head ###############")
    print(dataframe.head())
    print("############################ Tail ###############")
    print(dataframe.tail())
    print("############################ NA ###############")
    print(dataframe.isnull().sum())
    print("############################ Quantiles ###############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99,1]).T)

check_df(df_control)

check_df(df_test)

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df_control["group"]= "control"
df_test["group"]="test"

df=pd.concat([df_control,df_test], axis=0, ignore_index=False)
df.head()


##################################
# Görev 2: A/B Testinin Hipotezinin Tanımlanması
#################################

# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2 (Kontrol grubu ve test grubu satınalma ortalamaları arasında fark yoktur)
# H1 : M1!= M2 (Kontrol grubu ve test grubu satınalma ortalamaları arasında fark vardır)

# Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.

df.groupby("group").agg({"Purchase":"mean"})

##################################
# Görev 3: Hipotez Testinin Gerçekleştirilmesi
#################################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# -----------------------------------------------------------------------------------------------------------
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.
# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.

# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.5891   (HO REDDEDİLEMEZ.CONTROL GRUBUNUN DEĞERLERİ NORMAL DAĞILIM VARSAYIMINI SAĞLAMAKTADIR)

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.1541 (HO REDDEDİLEMEZ.TEST GRUBUNUN DEĞERLERİ NORMAL DAĞILIM VARSAYIMINI SAĞLAMAKTADIR)

# Normal dağılım sağlandığı için varyans homojenliği tesitine geçiyoruz.
# Sağlanmasaydı Nonparametrik test yapacaktık.
# Varyans Homojenliği :
# Varyans, verilerin aritmetik ortalamadan sapmalarınının karelerinin toplamıdır.Yani standart sapmanın karekök alınmış halidir.
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.1083   (HO REDDEDİLEMEZ.CONTROL VE TEST GRUBUNUN DEĞERLERİ VARYANS HOMOJENLİĞİ VARSAYIMINI SAĞLAMAKTADIR)
# Varyans Homojenliği sağlandığı için bağımsız iki örneklem t testine geçeceğiz.
# Eğer varyans homojenliği sağlanmasaydı sadece tek bir argümanın equal_var ı False olan değişkenimize varyans homojenliği için onu uygulayacaktık.


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

# Varyanslar sağlandığı için bağımsız iki örneklem t testi (parametrik test) yapılmaktadır.
# H0: M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (eşit değildir) (Kontrol grubu ve test grubu satın alma ortalamaları Arasında İstatistiksel Olarak Anl. Fark.vardır)
# M1 ve M2=Ana kitle ortalamasının temsilleri.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# p-value=0.3493  (HO REDDEDİLEMEZ.CONTROL VE TEST GRUBUNUN SATIN ALMA ORTALAMALARI ARASINDA İSTATİKSEL OLARAK ANLAMLI FARK YOKTUR)



##################################
# Görev 4: Sonuçların Analizi
#################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
# İlk önce iki gruba da normallik testi uygulanmıştır.İki grubunda normal dağılıma uyduğu gözlemlendiğinden
# İkinci varsayıma geçilerek varyansın homojenliği incelenmiştir.
# Varyanslar homojen çıktığından "Bağımsız İki Örneklem T Testi" uygulanmıştır.
# Uygulama sonucunda p-değerinin 0.05 ten büyük olduğu gözlemlenmiştir ve H0 hipotezi reddedilememiştir.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
#  Satın alma analamında anlamlı bir fark olmadığından müşteri iki yöntemden birini seçebilir.Fakat burada diğer istatistiklerdeki farklılıklar da önem arz edecektir.
#  Tıklanma,etkileşim,kazanç ve dönüşüm oranlarındaki farklılıklar değerlendirilip hangi yöntemin daha kazançlı olduğu tespit edilebilir.
#  Özellikle Facebook a tıklanma başına para ödendiği için hangi yöntemde tıklanma oranının daha düşük oldugu tespit edilir ve CTR oranına bakılabilir iki grup gözlenmeye devam edilir.

