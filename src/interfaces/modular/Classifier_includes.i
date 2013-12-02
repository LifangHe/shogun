%{
 #include <machine/Machine.h>
 #include <classifier/svm/GNPPSVM.h>
 #include <classifier/svm/GPBTSVM.h>
 #include <machine/DistanceMachine.h>
 #include <classifier/LDA.h>
 #include <classifier/svm/LibLinear.h>
 #include <classifier/svm/LibSVM.h>
 #include <classifier/svm/LibSVMOneClass.h>
 #include <machine/LinearMachine.h>
 #include <machine/OnlineLinearMachine.h>
 #include <classifier/LPBoost.h>
 #include <classifier/LPM.h>
 #include <classifier/svm/MPDSVM.h>
 #include <classifier/svm/OnlineSVMSGD.h>
 #include <classifier/svm/OnlineLibLinear.h>
 #include <classifier/Perceptron.h>
 #include <classifier/AveragedPerceptron.h>
 #include <classifier/svm/SVM.h>
 #include <classifier/svm/SVMLin.h>
 #include <machine/KernelMachine.h>
 #include <classifier/svm/SVMOcas.h>
 #include <classifier/svm/SVMSGD.h>
 #include <classifier/svm/SGDQN.h>
 #include <classifier/svm/WDSVMOcas.h>
 #include <classifier/PluginEstimate.h>
 #include <classifier/mkl/MKL.h>
 #include <classifier/mkl/MKLClassification.h>
 #include <classifier/mkl/MKLOneClass.h>
 #include <classifier/vw/VowpalWabbit.h>
 #include <classifier/svm/NewtonSVM.h>

#ifdef SHOGUN_USE_SVMLIGHT
 #include <classifier/svm/SVMLight.h>
 #include <classifier/svm/SVMLightOneClass.h>
#endif // SHOGUN_USE_SVMLIGHT
 #include <classifier/FeatureBlockLogisticRegression.h>
 #include <machine/DirectorLinearMachine.h>
 #include <machine/DirectorKernelMachine.h>
 #include <machine/BaggingMachine.h>
%}
