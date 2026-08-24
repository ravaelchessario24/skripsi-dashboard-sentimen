import re
import pandas as pd
import nltk

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def case_folding(teks):
    return str(teks).lower()

def cleansing(teks):
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'@\w+', '', teks)
    teks = re.sub(r'#\w+', '', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

kamus_df = pd.read_csv("kamus_alay.csv")

kamus_normalisasi = dict(
    zip(kamus_df["L1: slang"], kamus_df["formal"])
)
kamus_tambahan = {
    'diperbaiki': 'perbaiki',
    'diklinik': 'klinik',
    'dipakai': 'pakai',
    'diambil': 'ambil',
    'mau': 'ingin',
    'sayah': 'saya',
    'gua': 'saya',
    'gue': 'saya',
    'aku': 'saya',
    'gk': 'tidak',
    'ga': 'tidak',
    'gak': 'tidak',
    'kaga': 'tidak',
    'kagak': 'tidak',
    'tiap': 'setiap',
    'enggak': 'tidak',
    'gaada': 'tidak ada',
    'gabisa': 'tidak bisa',
    'kalo': 'kalau',
    'lagij': 'lagi',
    'mohon': 'tolong',
    'trus': 'terus',
    'truss': 'terus',
    'terooosss': 'terus',
    'udah': 'sudah',
    'udh': 'sudah',
    'tp': 'tapi',
    'krn': 'karena',
    'jg': 'juga',
    'yg': 'yang',
    'dgn': 'dengan',
    'utk': 'untuk',
    'sdh': 'sudah',
    'mapah': 'malah',
    'lg': 'lagi',
    'aj': 'saja',
    'aja': 'saja',
    'bener': 'benar',
    'beneran': 'benar',
    'pengen': 'ingin',
    'pgn': 'ingin',
    'emg': 'memang',
    'emang': 'memang',
    'knp': 'kenapa',
    'gmn': 'bagaimana',
    'blm': 'belum',
    'thx': 'terimakasih',
    'jdi': 'jadi',
    'sm': 'sama',
    'dr': 'dari',
    'pd': 'pada',
    'bs': 'bisa',
    'tdk': 'tidak',
    'engga': 'tidak',
    'ngga': 'tidak',
    'nggak': 'tidak',
    'mulu': 'terus',
    'erorr': 'error',
    'apk': 'aplikasi',
    'logut': 'logout',
    'eror': 'error',
    'no': 'nomor',
    'logij': 'login',
    'gagl': 'gagal',
    'dibutuhkn': 'dibutuhkan',
    'lemot': 'lambat',
    'lola': 'lambat',
    'loding': 'lambat',
    'loading': 'lambat',
    'loadingnya': 'lambat',
    'x': 'kali',
    'pasword': 'password',
    'oket': 'oke',
    'dg': 'dengan',
    'mobil': 'mobile',
    'utj': 'untuk',
    'rs': 'rumah sakit',
    'jaring': 'jaringan',
    'moga': 'semoga',
    'cuma': 'hanya',
    'd': 'di',
    'ny': 'nya',
    'bgt': 'banget',
    'sih': 'sih',
    'nya': 'nya',
    'lelet': 'lambat',
    'lemot': 'lambat',
    'log out': 'logout',
    'loug out': 'logout',
    'senditi': 'sendiri',
    'pembaru': 'pembaruan',
    'bukanya': 'bukannya',
    'bego': 'dongo',
    'ok': 'oke',
    'hp': 'ponsel',
    'captifa': 'captcha',
    'ribetkenapa': 'ribet kenapa',
    'lagigimana': 'lagi bagaimana',

    # typo
    'muas': 'puas',
    'trimakasih': 'terima kasih',
    'nomer': 'nomor',
    'rubah': 'ubah',
    'antre': 'antri',
    'erti': 'mengerti',

    # slang apresiasi
    'mantul': 'mantap',
    'mantab': 'mantap',
    'jos': 'bagus',
    'sip': 'bagus',
    'job': 'bagus',
    'nice': 'bagus',
    'best': 'terbaik',
    'satset': 'cepat',
    'gercep': 'cepat',
    'cuma': 'namun',

    # slang jawa
    'pol': 'sangat',
    'haturnuhun': 'terima kasih',

    # translate indonesia
    'app': 'aplikasi',
    'apl': 'aplikasi',
    'bug': 'error',
    'simple': 'sederhana',
    'simpel': 'sederhana',
    'good': 'bagus',

    # kata tidak baku
    'tau': 'tahu',
    'cek': 'periksa',
    'min': 'admin',
    'punya': 'milik',
}

kamus_gabungan = {**kamus_normalisasi, **kamus_tambahan}

def normalisasi(teks):
    if pd.isna(teks):
        return ""

    teks = str(teks).lower()

    frasa_khusus = {
        'log out': 'logout',
        'loug out': 'logout',
        'log in': 'login',
        'terima kasih': 'terimakasih'
    }
    for kata_lama, kata_baru in frasa_khusus.items():
        teks = teks.replace(kata_lama, kata_baru)

    kata_kata = teks.split()
    kata_normal = [kamus_gabungan.get(kata, kata) for kata in kata_kata]

    return ' '.join(kata_normal)

def tokenizing(teks):
    return word_tokenize(teks)

factory_stop = StopWordRemoverFactory()
stopword_list = set(factory_stop.get_stop_words())

whitelist = {
    # negasi
    'tidak', 'bukan', 'tanpa', 'belum', 'gagal', 'bisa', 'masih', 'padahal', 'malah',
    'justru', 'sendiri', 'masuk', 'keluar', 'terus', 'banyak', 'kenapa', 'sering', 'terbantu',
    'namun', 'setelah', 'bukannya',

    # negatif eksplisit
    'susah', 'sulit', 'buruk', 'jelek', 'kecewa',
    'marah', 'kesal', 'lemot', 'lambat', 'ribet',
    'kendala', 'error', 'kurang',

    # positif eksplisit
    'baik', 'bagus', 'puas', 'benar', 'senang',
    'siap', 'cukup', 'membantu', 'sangat',

    # kata positif
    'mudah', 'bantu', 'mantap', 'cepat', 'keren',
    'lancar', 'praktis', 'gampang', 'nyaman', 'lengkap',
    'ramah', 'sukses', 'aman', 'terbantu',

    # kata teknis
    'login', 'logout', 'verifikasi', 'otp', 'antri',
    'layan', 'manfaat', 'akses', 'sehat', 'faskes',
    'wajah', 'jaringan',
}

stopword_tambahan = {
    'nya', 'yg', 'dgn', 'utk', 'jg', 'sdh', 'krn', 'sih', 'deh', 'dong', 'nih', 'loh', 'lah', 'kah',
    'banget', 'aaahhhh', 'hadehhhh', 'wkwk', 'haha', 'hehe', 'wkwkwk', 'hahahaha', 'lo', 'lu', 'out',
    'bgt', 'semalem', 'the', 'kak', 'kakak', 'ku', 'min', 'dll', 'via', 'kis', 'wa', 'moga', 'serta',
    'telah', 'pokok', 'biar', 'bikin', 'indonesia', 'rakyat',
    'di', 'in', 'pas', 'gini', 'itu', 'ini'
}

stopword_list = set(factory_stop.get_stop_words()) - whitelist | stopword_tambahan

def stopword_removal(tokens):
    return [kata for kata in tokens if kata not in stopword_list and len(kata) > 1]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

kata_lindungi = {
    'perbaiki', 'memperbaiki', 'mempersulit', 'memasuki', 'terbaik', 'kekurangan', 'buatan',
    'membantu', 'mempermudah', 'memudahkan', 'memuaskan', 'bermanfaat', 'terbuka', 'pertahankan',
    'berguna', 'pelayanan', 'layanan', 'kesehatan', 'menyusahkan', 'mengesalkan',
    'aplikasi', 'verifikasi', 'jaringan', 'antrian', 'tingkatkan', 'setelah', 'permintaan', 'pengembang',
    'antrean', 'pengguna', 'peserta', 'giliran', 'alhamdulillah', 'pembaruan', 'tangani', 'ditangani',
    'semoga', 'pemerintah', 'keluarga', 'berobat', 'terbantu', 'logout', 'bukannya', 'pendaftaran',

    # TAMBAHAN HASIL VERIFIKASI

    'perbaikan',      
    'masukan',        
    'keterangan',     

    'dipersulit',     
    'persulit',       
    'menyulitkan',    
    'bermasalah',     
    'gangguan',       

    'terbantu',  
    'permudah',  
    'pelayanannya', 
}

@lru_cache(maxsize=None)
def _stem_cached(kata):
    return stemmer.stem(kata)

def selective_stemming(tokens):
    result = []
    for kata in tokens:
        if kata in kata_lindungi:
            result.append(kata)
        else:
            result.append(_stem_cached(kata))
    return result

def preprocessing(teks):

    teks = case_folding(teks)
    teks = cleansing(teks)
    teks = normalisasi(teks)

    tokens = tokenizing(teks)
    tokens = stopword_removal(tokens)
    tokens = selective_stemming(tokens)

    return " ".join(tokens).strip()

if __name__ == "__main__":
    contoh = "Aplikasi jelek, tidak bisa login!!!"

    hasil = preprocessing(contoh)

    print("Input       :", contoh)
    print("Preprocessing:", hasil)