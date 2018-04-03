from lbp_score import LBP
import cv2

'''
实验：用来才是哪一种lbp直方图的距离判据更加好。
'''
methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_BHATTACHARYYA,
           cv2.HISTCMP_CHISQR, cv2.HISTCMP_CHISQR_ALT,
           cv2.HISTCMP_HELLINGER, cv2.HISTCMP_INTERSECT,
           cv2.HISTCMP_KL_DIV]
methods_name = ['cv2.HISTCMP_CORREL', 'cv2.HISTCMP_BHATTACHARYYA',
                'cv2.HISTCMP_CHISQR', 'cv2.HISTCMP_CHISQR_ALT',
                'cv2.HISTCMP_HELLINGER', 'cv2.HISTCMP_INTERSECT',
                'cv2.HISTCMP_KL_DIV']
imgs_path = ['datasets/plates/testA/Cr_4.jpg',
             'datasets/plates/testA/In_66.jpg',
             'datasets/plates/testA/Pa_11.jpg',
             'datasets/plates/testA/PS_266.jpg',
             'datasets/plates/testA/RS_91.jpg',
             'datasets/plates/testA/Sc_218.jpg']
def start_compare(method):
    allstr = ''
    print(method)
    for histA in images_hist:
        for histB in images_hist:
            re = cv2.compareHist(histA, histB, method=method)
            print(round(re, 3), end=',')
            allstr += str(round(re, 3))
            allstr += ','
        allstr += '\n'
        print('')
    return allstr
if __name__ == "__main__":

    images_hist = []
    for imgA in imgs_path:
        img = cv2.imread(imgA,flags=cv2.IMREAD_GRAYSCALE)
        l1 = LBP(img)
        # hist = l1.get_lbp_hist()
        hist = l1.get_uniform_lbp_hist()
        images_hist.append(hist)

    final_text = ''
    for i,m in enumerate(methods):
        data = methods_name[i]+'\n'
        data += 'Cr_4,In_66,Pa_11,PS_266,RS_91,Sc_218\n'
        data += start_compare(method=m)
        final_text+=data
    f = open('lbp_result.csv',mode='w')
    f.write(final_text)
    f.close()
