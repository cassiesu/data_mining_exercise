import sys
import math
import numpy as np
import scipy
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import SkewedChi2Sampler, RBFSampler, AdditiveChi2Sampler, Nystroem

def transform(x_original, make_np=True):
    orig = x_original
    MEAN = [ 0.00213536,  0.00324656,  0.00334724,  0.00175428,  0.00349227,
          0.0035413 ,  0.00188289,  0.00216241,  0.00184026,  0.00351317,
          0.00520942,  0.00450718,  0.00346782,  0.00300477,  0.00223811,
          0.00180039,  0.00216675,  0.00381716,  0.00258565,  0.00291358,
          0.00616643,  0.00237084,  0.00440006,  0.00729192,  0.00369302,
          0.00058215,  0.00312047,  0.00629086,  0.00184585,  0.0018266 ,
          0.00329771,  0.00352135,  0.00246634,  0.00261958,  0.00357113,
          0.00307333,  0.00211512,  0.00125184,  0.00212255,  0.00307451,
          0.00171408,  0.0126576 ,  0.00252346,  0.00528872,  0.0026387 ,
          0.00283739,  0.00394586,  0.00207473,  0.00307515,  0.002017  ,
          0.00408066,  0.00185709,  0.00316201,  0.00349098,  0.00415104,
          0.00348125,  0.00069981,  0.00128145,  0.0023404 ,  0.00396659,
          0.00240324,  0.01251434,  0.00125352,  0.00266113,  0.00435828,
          0.00066137,  0.00221134,  0.00083185,  0.00278664,  0.00118505,
          0.00335414,  0.00340527,  0.0026939 ,  0.00096786,  0.00214149,
          0.0026521 ,  0.00155538,  0.00300255,  0.0040405 ,  0.00275396,
          0.00077404,  0.00257667,  0.00268743,  0.00279948,  0.0018655 ,
          0.00239569,  0.0032419 ,  0.00288355,  0.00123361,  0.00220135,
          0.0021836 ,  0.00225123,  0.00366629,  0.00279189,  0.00058814,
          0.00310452,  0.00276981,  0.00128716,  0.00074161,  0.00358908,
          0.003292  ,  0.00233592,  0.00317694,  0.00381526,  0.00269197,
          0.00098085,  0.00231831,  0.00133682,  0.00460957,  0.00387842,
          0.0004473 ,  0.0015644 ,  0.00247717,  0.00179484,  0.00281831,
          0.00053689,  0.00415889,  0.00232736,  0.00361601,  0.00192624,
          0.00224487,  0.00210838,  0.00140079,  0.00608319,  0.00211861,
          0.00230604,  0.00124033,  0.0029389 ,  0.00227564,  0.00086638,
          0.0035496 ,  0.00228789,  0.00361703,  0.00270277,  0.00196611,
          0.00206865,  0.00146788,  0.00019011,  0.00222272,  0.00351472,
          0.00305718,  0.00239471,  0.00040766,  0.00299186,  0.00368983,
          0.00244158,  0.00084154,  0.00109796,  0.00278565,  0.00135904,
          0.00424855,  0.00323784,  0.00255397,  0.00234946,  0.00210558,
          0.00291688,  0.00172516,  0.00284473,  0.00308164,  0.00316225,
          0.0041659 ,  0.00055891,  0.00303591,  0.00028217,  0.00261526,
          0.00196658,  0.00264379,  0.00018002,  0.00227361,  0.00190785,
          0.00344782,  0.00305479,  0.00057851,  0.00115452,  0.00365707,
          0.0009598 ,  0.00184313,  0.00286183,  0.00400594,  0.0003848 ,
          0.00086102,  0.00277779,  0.00214625,  0.00329827,  0.00129511,
          0.00114751,  0.00249452,  0.00236266,  0.00353646,  0.00319208,
          0.00540883,  0.00323167,  0.00299791,  0.00025745,  0.00227873,
          0.00228826,  0.0040653 ,  0.00238598,  0.00483883,  0.00054585,
          0.00091663,  0.00037232,  0.0008229 ,  0.00073563,  0.00283771,
          0.0035899 ,  0.00578833,  0.0032107 ,  0.0014048 ,  0.00401052,
          0.002748  ,  0.00229416,  0.00130351,  0.00308403,  0.00146506,
          0.00188529,  0.00236308,  0.00259649,  0.00185155,  0.00230195,
          0.00421584,  0.00231917,  0.00227335,  0.00296253,  0.00077996,
          0.0001668 ,  0.00069015,  0.00220702,  0.00238395,  0.00034903,
          0.00303323,  0.00407338,  0.00178655,  0.00456887,  0.00254606,
          0.00215019,  0.00306377,  0.00134979,  0.00112832,  0.00350681,
          0.00253643,  0.00431348,  0.00094915,  0.00150396,  0.00043838,
          0.00207101,  0.00301119,  0.00057716,  0.00062709,  0.00543404,
          0.00061686,  0.00237189,  0.00522715,  0.00321869,  0.00172645,
          0.00244482,  0.00334951,  0.00183201,  0.00038157,  0.0023022 ,
          0.00418559,  0.00329119,  0.00411452,  0.00089033,  0.00283673,
          0.00210368,  0.00222242,  0.00213262,  0.0033576 ,  0.00250707,
          0.00423595,  0.00237407,  0.00127654,  0.00387341,  0.00216695,
          0.00325004,  0.00246333,  0.00396034,  0.0031676 ,  0.00354552,
          0.00227099,  0.00205363,  0.00128859,  0.00290737,  0.00301655,
          0.00319576,  0.00072449,  0.00230528,  0.00326406,  0.00283315,
          0.00338869,  0.00212552,  0.00135612,  0.00250613,  0.00045907,
          0.0014009 ,  0.00177951,  0.00042544,  0.00073249,  0.00303487,
          0.0013664 ,  0.00248306,  0.00025601,  0.00435174,  0.00443799,
          0.00479944,  0.0009997 ,  0.00275155,  0.00286969,  0.00244896,
          0.00177604,  0.00278218,  0.00078876,  0.00142078,  0.00186949,
          0.0018215 ,  0.0027254 ,  0.00316367,  0.00192957,  0.00176559,
          0.00289111,  0.00048977,  0.00411342,  0.00130383,  0.00250934,
          0.00324275,  0.00159243,  0.00334068,  0.00324279,  0.00158259,
          0.00041714,  0.00161102,  0.00145149,  0.00222112,  0.00296289,
          0.00282892,  0.00123731,  0.00281891,  0.00016613,  0.0014267 ,
          0.00262089,  0.00367506,  0.00281706,  0.00318947,  0.00090315,
          0.00230826,  0.00310803,  0.00889549,  0.00197781,  0.00160006,
          0.00307063,  0.00176858,  0.00252353,  0.00141795,  0.00047073,
          0.00241224,  0.00165672,  0.00138939,  0.00257068,  0.00148445,
          0.00193734,  0.004368  ,  0.00247817,  0.00249266,  0.00329317,
          0.00078468,  0.00045822,  0.00259324,  0.00298367,  0.00335009,
          0.00307879,  0.00325237,  0.00254531,  0.00749495,  0.0026701 ,
          0.00100689,  0.00184948,  0.00317616,  0.00255977,  0.00112342,
          0.00165774,  0.00227449,  0.00064219,  0.00269639,  0.00114312,
          0.00203549,  0.00064574,  0.00130932,  0.00304631,  0.00131053,
          0.00174587,  0.0027975 ,  0.00461148,  0.0015227 ,  0.0027072 ,
          0.00210673,  0.00323388,  0.00028426,  0.00113429,  0.00315131]

    VAR = [  3.87111312e-06,   1.29838726e-05,   1.23895436e-05,
           5.11051819e-06,   1.87834728e-05,   5.81101229e-05,
           1.22431672e-05,   3.14238203e-06,   6.15186426e-06,
           1.16054974e-05,   2.61629851e-05,   1.51823678e-05,
           3.20501352e-05,   6.75625364e-06,   6.90383937e-06,
           7.10772563e-06,   3.93108356e-06,   1.38147699e-05,
           9.45390664e-06,   6.18869987e-06,   1.23460353e-03,
           5.15741591e-06,   1.27185867e-05,   7.62148434e-05,
           9.61369316e-06,   3.59794999e-06,   4.49714597e-05,
           1.15313013e-04,   2.51027515e-06,   3.23518027e-06,
           1.15175054e-05,   5.55007797e-05,   3.61287015e-06,
           4.24901217e-06,   1.57731133e-05,   8.83739880e-06,
           4.11832891e-06,   4.51594425e-06,   5.66233716e-06,
           2.76312055e-05,   3.10286633e-05,   2.06523833e-04,
           4.99679342e-06,   3.59423460e-05,   5.53408014e-06,
           5.02979264e-06,   2.29845095e-05,   3.52580303e-06,
           4.74110466e-06,   2.77776825e-06,   1.15279947e-05,
           4.78634098e-06,   8.24242505e-06,   1.65141090e-05,
           1.84669015e-05,   1.65851869e-05,   9.69125917e-07,
           4.07269628e-06,   4.79411492e-06,   7.95185399e-06,
           6.05491604e-06,   2.30133633e-04,   2.43045915e-06,
           9.99138675e-06,   1.61846281e-05,   1.36250194e-06,
           3.83900385e-06,   4.03501076e-06,   4.49190746e-06,
           2.20133970e-06,   1.40571788e-05,   1.23973871e-05,
           1.91642968e-05,   1.83384119e-06,   3.55110501e-06,
           6.38707023e-06,   7.58389225e-06,   9.66052931e-06,
           1.33459561e-05,   6.01834583e-06,   1.75975058e-06,
           9.93625536e-06,   5.57880408e-06,   5.20632392e-06,
           2.63891241e-06,   4.96341232e-06,   1.35361419e-05,
           5.09588225e-06,   2.13213362e-06,   3.67884149e-06,
           4.02580880e-06,   3.36118966e-06,   1.23913905e-05,
           1.19327162e-05,   1.33013390e-06,   1.56844681e-05,
           5.05235129e-06,   3.27510379e-06,   4.18496352e-06,
           1.32615022e-05,   8.00089632e-06,   5.24889508e-06,
           7.61725520e-06,   2.45732025e-05,   4.73942392e-06,
           3.26874106e-06,   4.19502445e-06,   4.67408597e-06,
           4.07529951e-05,   1.85623369e-05,   1.42640177e-06,
           9.02420306e-06,   3.99465979e-06,   2.91695819e-06,
           7.51525182e-06,   3.28339831e-06,   9.23579413e-06,
           8.82938566e-06,   1.67017625e-05,   7.18046179e-06,
           6.67502140e-06,   4.53568390e-06,   4.59241197e-06,
           9.71055426e-05,   4.06108283e-06,   3.21309715e-06,
           2.83145362e-06,   1.30979068e-05,   4.30934096e-06,
           1.33494112e-06,   1.23067054e-05,   4.55467345e-06,
           4.16151366e-05,   4.39300907e-06,   3.81081336e-06,
           3.57599046e-06,   2.44792045e-06,   1.04884156e-06,
           5.66646773e-06,   1.38454953e-05,   7.03958785e-06,
           7.96561298e-06,   1.15832827e-06,   5.34098000e-06,
           1.08664502e-05,   5.33706713e-06,   1.58029233e-06,
           4.16948014e-06,   1.10410603e-05,   3.08923185e-06,
           3.60056097e-05,   1.35575315e-05,   7.21297470e-06,
           5.46186866e-06,   3.83067878e-06,   4.93382163e-06,
           8.74249160e-06,   6.95763983e-06,   8.57639945e-06,
           1.99238085e-05,   2.06143616e-05,   4.15158574e-06,
           6.98539924e-06,   7.29978665e-07,   1.05324242e-05,
           4.03610511e-06,   4.54024757e-06,   1.12380259e-06,
           7.25149490e-06,   4.68609708e-06,   4.47583007e-05,
           5.73128000e-06,   1.55383559e-06,   6.10201277e-06,
           1.56226083e-05,   2.07417481e-06,   3.92362694e-06,
           5.07511158e-06,   1.91527526e-05,   1.23196439e-06,
           2.78105795e-06,   6.20886459e-06,   9.77619759e-06,
           4.54569998e-05,   3.69801329e-06,   3.90055801e-06,
           8.95043365e-06,   4.62714915e-06,   8.59072207e-06,
           7.93476416e-06,   2.94461267e-05,   1.27513460e-05,
           6.37168538e-06,   1.42869302e-06,   3.88169829e-06,
           3.73479924e-06,   3.41961106e-05,   5.99249536e-06,
           3.52894229e-05,   3.60535269e-06,   1.97432492e-06,
           1.08726206e-06,   6.34745318e-06,   1.85853697e-06,
           4.88355657e-06,   1.45421337e-05,   4.71209759e-05,
           9.75886239e-06,   1.92188254e-06,   2.44175182e-05,
           6.48665880e-06,   3.77833988e-06,   4.94021824e-06,
           1.11375076e-05,   2.48913056e-06,   7.50221434e-06,
           7.71706724e-06,   4.40449246e-06,   5.01260110e-06,
           7.55913298e-06,   9.61114153e-06,   4.71524238e-06,
           5.71612330e-06,   5.35067657e-06,   1.24371020e-06,
           1.05315411e-06,   3.93981671e-06,   4.10917913e-06,
           4.50131192e-06,   1.41029887e-06,   5.21404239e-06,
           3.10300539e-05,   2.86295992e-06,   3.14574375e-05,
           4.13089781e-06,   3.94511845e-06,   5.21837923e-06,
           1.86040011e-06,   4.33877122e-06,   6.79169351e-06,
           7.34233345e-06,   2.46684357e-05,   6.04518227e-06,
           3.50075336e-06,   1.22008735e-06,   3.82670787e-06,
           1.29928488e-05,   1.30317263e-06,   1.82923403e-06,
           1.68159694e-04,   1.39570985e-06,   6.82018782e-06,
           2.77705938e-05,   5.50219803e-06,   6.94297855e-06,
           5.56691651e-06,   4.40913139e-05,   8.64954832e-06,
           1.13623461e-06,   3.91895303e-06,   2.90528320e-05,
           8.95829181e-06,   2.13802762e-05,   1.45383845e-06,
           2.19748855e-05,   2.92403666e-06,   4.11580346e-06,
           3.79422424e-06,   1.01354981e-05,   1.12666398e-05,
           2.12954971e-05,   4.73278161e-06,   2.26826965e-06,
           2.45301255e-05,   5.86185180e-06,   6.92235736e-06,
           8.42678526e-06,   2.47795958e-05,   6.25412728e-06,
           1.41974527e-05,   3.95337688e-06,   7.16912125e-06,
           2.00884144e-06,   2.00349034e-05,   5.97662651e-06,
           3.01450892e-05,   4.63002816e-06,   4.09857661e-06,
           1.23373959e-05,   5.62286236e-06,   1.23868932e-05,
           7.79128188e-06,   4.02737664e-06,   4.26867074e-06,
           1.30633550e-06,   2.16092242e-06,   2.53344988e-06,
           1.55130629e-06,   1.20587686e-06,   8.47719131e-06,
           1.72865161e-06,   8.85885938e-06,   1.36250583e-06,
           3.02467214e-05,   2.85941868e-05,   1.68684969e-05,
           2.17024274e-06,   9.09429716e-06,   1.12517072e-05,
           5.39997088e-06,   3.16738113e-06,   7.44227101e-06,
           1.39521345e-06,   1.80325624e-06,   3.23437991e-06,
           4.12906812e-06,   6.51981136e-06,   7.28606378e-06,
           4.44469608e-06,   4.00705337e-06,   1.34244753e-05,
           1.34953189e-06,   3.86701616e-05,   4.30733919e-06,
           4.29618197e-06,   1.67568650e-05,   5.39451612e-06,
           8.50733433e-06,   1.04900918e-05,   4.68246794e-06,
           2.92591087e-06,   2.54589900e-06,   6.68970689e-06,
           3.68698856e-06,   5.70542637e-06,   1.57329410e-05,
           3.45199222e-06,   7.27799975e-06,   8.64176250e-07,
           5.59882582e-06,   4.16052401e-06,   1.73753080e-05,
           7.85748797e-06,   6.46626446e-06,   2.23241624e-06,
           6.79217908e-06,   6.18545939e-06,   5.41203600e-04,
           2.75355566e-06,   5.01654998e-06,   9.55004050e-06,
           3.36241075e-06,   4.95540827e-06,   4.38650100e-06,
           2.19975452e-06,   4.99878215e-06,   2.08615031e-06,
           6.57349770e-06,   6.07825138e-06,   1.82116637e-05,
           3.98356104e-06,   3.02862803e-05,   1.45275531e-05,
           1.80111343e-05,   1.81263109e-05,   1.37630960e-06,
           1.01588605e-06,   1.09961427e-05,   7.09189456e-06,
           8.63553483e-06,   1.28377215e-05,   1.15539997e-05,
           4.30247032e-06,   3.69651334e-05,   1.13411365e-05,
           1.43191945e-06,   2.76733205e-06,   7.03730009e-06,
           4.93027252e-06,   2.72768641e-06,   3.15867713e-06,
           3.51786262e-06,   1.33668414e-06,   5.15268762e-06,
           2.24808552e-06,   3.91888753e-06,   1.96848802e-06,
           5.96948656e-06,   6.72807533e-06,   2.52024742e-06,
           4.64795350e-06,   6.00152269e-06,   4.42994740e-05,
           2.59223022e-06,   4.76032620e-06,   3.15249648e-06,
           1.02942457e-05,   7.54992395e-07,   2.48130225e-06,
           5.97253972e-06];

    x_original = np.array(x_original)
    x_original -= MEAN
    x_original /= VAR

    def extend_x(arr, additions=True, extension=True):
        if extension:
            x.extend(arr)
        if additions:
            x.append(scipy.std(arr))
            x.append(scipy.var(arr))
            x.append(sum(arr) / len(arr))
            x.append(sum(np.abs(arr)) / len(arr))
            x.append(min(arr))
            x.append(max(arr))
            x.append(scipy.mean(arr))
            x.append(scipy.median(arr))

    x = []

    extend_x(x_original)
    extend_x(np.abs(x_original))
    # extend_x(np.sqrt(np.abs(x_original)))

    # sampler1 = SkewedChi2Sampler(skewedness=0.022, n_components=50, random_state=1)
    # zzz1 = sampler1.fit_transform(np.abs(np.array(orig)))[0]

    # sampler2 = SkewedChi2Sampler(skewedness=8.5, n_components=50, random_state=1)
    # zzz2 = sampler2.fit_transform(np.abs(np.array(x)))[0]

    sampler3 = RBFSampler(gamma=0.0025, random_state=2, n_components=20)
    zzz3 = sampler3.fit_transform(np.array(x))[0]


    extend_x(list(zzz1))
    extend_x(list(zzz2))
    extend_x(list(zzz3))

    if make_np:
        return np.array(x)
    
    return x


if __name__ == "__main__":

    clf = LinearSVC(dual=False, C=1.1, loss='l2', penalty='l2', tol=1e-4)
    # clf = LinearSVC(dual=True, C=1.1, loss='l1', penalty='l2')
    # clf = linear_model.SGDClassifier(fit_intercept=False, alpha=0.00005, loss='hinge')

    X = []
    Y = []

    reader = sys.stdin
    for line in reader:
        line = line.strip().split(' ')
        Y.append(int(line.pop(0)))
        X.append(transform(map(float, line), False))

    X = np.array(X)
    Y = np.array(Y)

    def fit(X, Y):
        z = clf.fit(X, Y)
        print("1\t%s" % ",".join(map(str, z.coef_[0])))
            
    fit(X, Y)









