# vMF statistic (docs/vmf_plan.md), K = 100

   corpus            model   analysis feature     est  auc_real  auc_ctrl      lo      hi  n_draws
    nobel   openai-3-small       main vmf2_r5  0.4060       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small       main vmf2_r3  0.4336       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small       main vmf2_r8  0.3974       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small       main    eig2  0.6614       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small  fixed_all vmf2_r5  0.1516    0.3779    0.2262  0.1119  0.1809      1.0
    nobel   openai-3-small  fixed_all    eig2  0.1106    0.3015    0.1909  0.0547  0.1725      1.0
    nobel   openai-3-small fixed_lodo vmf2_r5  0.1468    0.3621    0.2153  0.1057  0.1868     20.0
    nobel   openai-3-small fixed_lodo    eig2  0.1275    0.3160    0.1885  0.0693  0.1755     20.0
    nobel   openai-3-large       main vmf2_r5  0.4234       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large       main vmf2_r3  0.4233       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large       main vmf2_r8  0.4392       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large       main    eig2  0.7533       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large  fixed_all vmf2_r5  0.0836    0.3838    0.3002  0.0180  0.1290      1.0
    nobel   openai-3-large  fixed_all    eig2  0.0844    0.3092    0.2248  0.0315  0.1381      1.0
    nobel   openai-3-large fixed_lodo vmf2_r5  0.0579    0.3687    0.3108  0.0146  0.1020     20.0
    nobel   openai-3-large fixed_lodo    eig2  0.1089    0.3371    0.2283  0.0612  0.1530     20.0
    nobel       vertex-005       main vmf2_r5  0.4913       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005       main vmf2_r3  0.5075       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005       main vmf2_r8  0.4762       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005       main    eig2  0.6588       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005  fixed_all vmf2_r5 -0.0723    0.3884    0.4607 -0.1392 -0.0081      1.0
    nobel       vertex-005  fixed_all    eig2 -0.1949    0.1661    0.3610 -0.2308 -0.1551      1.0
    nobel       vertex-005 fixed_lodo vmf2_r5 -0.0873    0.3917    0.4790 -0.1377 -0.0297     20.0
    nobel       vertex-005 fixed_lodo    eig2 -0.1681    0.1979    0.3660 -0.2092 -0.1241     20.0
    nobel modernbert-embed       main vmf2_r5  0.5158       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed       main vmf2_r3  0.5816       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed       main vmf2_r8  0.4991       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed       main    eig2  0.7356       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed  fixed_all vmf2_r5  0.0963    0.4328    0.3365  0.0486  0.1398      1.0
    nobel modernbert-embed  fixed_all    eig2  0.1653    0.5492    0.3839  0.1198  0.2044      1.0
    nobel modernbert-embed fixed_lodo vmf2_r5  0.1134    0.4333    0.3199  0.0717  0.1530     20.0
    nobel modernbert-embed fixed_lodo    eig2  0.1385    0.5297    0.3912  0.0988  0.1818     20.0
    nobel         bge-base       main vmf2_r5  0.4781       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base       main vmf2_r3  0.4850       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base       main vmf2_r8  0.4862       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base       main    eig2  0.7015       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base  fixed_all vmf2_r5  0.2522    0.4671    0.2149  0.1964  0.2993      1.0
    nobel         bge-base  fixed_all    eig2  0.0240    0.3696    0.3457 -0.0130  0.0597      1.0
    nobel         bge-base fixed_lodo vmf2_r5  0.2189    0.4259    0.2070  0.1564  0.2709     20.0
    nobel         bge-base fixed_lodo    eig2  0.0193    0.3705    0.3512 -0.0208  0.0655     20.0
    nobel       gemini-001       main vmf2_r5  0.3874       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001       main vmf2_r3  0.4181       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001       main vmf2_r8  0.3756       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001       main    eig2  0.3014       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001  fixed_all vmf2_r5  0.1138    0.3011    0.1873  0.0515  0.1688      1.0
    nobel       gemini-001  fixed_all    eig2  0.0399    0.0494    0.0095  0.0228  0.0580      1.0
    nobel       gemini-001 fixed_lodo vmf2_r5  0.0826    0.2771    0.1945  0.0142  0.1392     20.0
    nobel       gemini-001 fixed_lodo    eig2  0.0533    0.0612    0.0078  0.0338  0.0774     20.0
     wiki   openai-3-small       main vmf2_r5  0.7778       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small       main vmf2_r3  0.7910       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small       main vmf2_r8  0.7733       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small       main    eig2  0.8490       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small  fixed_all vmf2_r5  0.1956    0.5312    0.3356  0.1717  0.2246      1.0
     wiki   openai-3-small  fixed_all    eig2  0.0596    0.2127    0.1531  0.0115  0.1002      1.0
     wiki   openai-3-small fixed_lodo vmf2_r5  0.1665    0.4845    0.3180  0.1292  0.2075     20.0
     wiki   openai-3-small fixed_lodo    eig2  0.0795    0.2351    0.1556  0.0393  0.1142     20.0
     wiki   openai-3-large       main vmf2_r5  0.7815       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large       main vmf2_r3  0.8098       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large       main vmf2_r8  0.7547       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large       main    eig2  0.9036       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large  fixed_all vmf2_r5  0.1217    0.5382    0.4166  0.0966  0.1581      1.0
     wiki   openai-3-large  fixed_all    eig2  0.0604    0.2406    0.1803  0.0230  0.0889      1.0
     wiki   openai-3-large fixed_lodo vmf2_r5  0.0508    0.4519    0.4011  0.0170  0.0856     20.0
     wiki   openai-3-large fixed_lodo    eig2  0.0552    0.2367    0.1814  0.0136  0.0876     20.0
     wiki       vertex-005       main vmf2_r5  0.8830       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005       main vmf2_r3  0.8111       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005       main vmf2_r8  0.9145       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005       main    eig2  0.9498       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005  fixed_all vmf2_r5  0.1032    0.5545    0.4513  0.0729  0.1386      1.0
     wiki       vertex-005  fixed_all    eig2  0.0567    0.3835    0.3268  0.0142  0.0888      1.0
     wiki       vertex-005 fixed_lodo vmf2_r5  0.1024    0.5507    0.4483  0.0714  0.1345     20.0
     wiki       vertex-005 fixed_lodo    eig2  0.0331    0.3627    0.3296 -0.0003  0.0627     20.0
     wiki modernbert-embed       main vmf2_r5  0.7516       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed       main vmf2_r3  0.7428       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed       main vmf2_r8  0.7652       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed       main    eig2  0.8385       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed  fixed_all vmf2_r5  0.0599    0.4016    0.3417  0.0109  0.1061      1.0
     wiki modernbert-embed  fixed_all    eig2  0.0399    0.2782    0.2382  0.0103  0.0671      1.0
     wiki modernbert-embed fixed_lodo vmf2_r5  0.0438    0.3765    0.3327 -0.0059  0.0967     20.0
     wiki modernbert-embed fixed_lodo    eig2  0.0322    0.2796    0.2473 -0.0089  0.0669     20.0
     wiki         bge-base       main vmf2_r5  0.7491       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base       main vmf2_r3  0.7491       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base       main vmf2_r8  0.7552       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base       main    eig2  0.7800       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base  fixed_all vmf2_r5  0.0428    0.5188    0.4760 -0.0009  0.0851      1.0
     wiki         bge-base  fixed_all    eig2  0.0028    0.2648    0.2620 -0.0311  0.0310      1.0
     wiki         bge-base fixed_lodo vmf2_r5  0.0381    0.5306    0.4925 -0.0042  0.0858     20.0
     wiki         bge-base fixed_lodo    eig2 -0.0021    0.2702    0.2723 -0.0396  0.0326     20.0
     wiki       gemini-001       main vmf2_r5  0.6388       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001       main vmf2_r3  0.6732       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001       main vmf2_r8  0.6072       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001       main    eig2  0.5327       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001  fixed_all vmf2_r5  0.1019    0.4389    0.3371  0.0659  0.1357      1.0
     wiki       gemini-001  fixed_all    eig2  0.0626    0.1080    0.0453  0.0275  0.0964      1.0
     wiki       gemini-001 fixed_lodo vmf2_r5  0.1324    0.4634    0.3310  0.0888  0.1734     20.0
     wiki       gemini-001 fixed_lodo    eig2  0.0499    0.0966    0.0467  0.0181  0.0874     20.0
gutenberg   openai-3-small       main vmf2_r5  0.4340       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small       main vmf2_r3  0.4594       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small       main vmf2_r8  0.4235       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small       main    eig2  0.5225       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small  fixed_all vmf2_r5 -0.0216    0.3776    0.3991 -0.0568  0.0103      1.0
gutenberg   openai-3-small  fixed_all    eig2  0.0192    0.2883    0.2691 -0.0123  0.0454      1.0
gutenberg   openai-3-small fixed_lodo vmf2_r5 -0.0098    0.3986    0.4084 -0.0358  0.0160     20.0
gutenberg   openai-3-small fixed_lodo    eig2  0.0112    0.3021    0.2909 -0.0166  0.0385     20.0
gutenberg   openai-3-large       main vmf2_r5  0.4393       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large       main vmf2_r3  0.4592       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large       main vmf2_r8  0.4369       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large       main    eig2  0.5553       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large  fixed_all vmf2_r5 -0.0213    0.3839    0.4052 -0.0609  0.0109      1.0
gutenberg   openai-3-large  fixed_all    eig2 -0.0007    0.2828    0.2835 -0.0401  0.0316      1.0
gutenberg   openai-3-large fixed_lodo vmf2_r5 -0.0109    0.4018    0.4127 -0.0434  0.0226     20.0
gutenberg   openai-3-large fixed_lodo    eig2 -0.0074    0.2959    0.3033 -0.0464  0.0286     20.0
gutenberg       vertex-005       main vmf2_r5  0.5590       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005       main vmf2_r3  0.5724       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005       main vmf2_r8  0.5464       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005       main    eig2  0.5321       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005  fixed_all vmf2_r5  0.0088    0.4464    0.4376 -0.0330  0.0541      1.0
gutenberg       vertex-005  fixed_all    eig2 -0.0256    0.3024    0.3280 -0.0552 -0.0001      1.0
gutenberg       vertex-005 fixed_lodo vmf2_r5 -0.0223    0.4197    0.4420 -0.0546  0.0114     20.0
gutenberg       vertex-005 fixed_lodo    eig2 -0.0603    0.2805    0.3408 -0.0911 -0.0318     20.0
gutenberg modernbert-embed       main vmf2_r5  0.4274       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed       main vmf2_r3  0.4625       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed       main vmf2_r8  0.4203       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed       main    eig2  0.5242       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed  fixed_all vmf2_r5 -0.0275    0.4106    0.4381 -0.0555 -0.0006      1.0
gutenberg modernbert-embed  fixed_all    eig2  0.0162    0.4411    0.4250 -0.0068  0.0357      1.0
gutenberg modernbert-embed fixed_lodo vmf2_r5 -0.0494    0.3925    0.4419 -0.0807 -0.0249     20.0
gutenberg modernbert-embed fixed_lodo    eig2 -0.0005    0.4290    0.4296 -0.0208  0.0161     20.0
gutenberg         bge-base       main vmf2_r5  0.4747       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base       main vmf2_r3  0.5123       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base       main vmf2_r8  0.4681       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base       main    eig2  0.5725       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base  fixed_all vmf2_r5  0.0070    0.4219    0.4148 -0.0267  0.0426      1.0
gutenberg         bge-base  fixed_all    eig2  0.0231    0.3848    0.3617 -0.0038  0.0475      1.0
gutenberg         bge-base fixed_lodo vmf2_r5  0.0161    0.4330    0.4169 -0.0128  0.0447     20.0
gutenberg         bge-base fixed_lodo    eig2  0.0158    0.3785    0.3627 -0.0077  0.0360     20.0
gutenberg       gemini-001       main vmf2_r5  0.4276       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001       main vmf2_r3  0.4695       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001       main vmf2_r8  0.4073       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001       main    eig2  0.3203       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001  fixed_all vmf2_r5  0.0509    0.3736    0.3227  0.0129  0.0882      1.0
gutenberg       gemini-001  fixed_all    eig2  0.0724    0.2115    0.1392  0.0334  0.1085      1.0
gutenberg       gemini-001 fixed_lodo vmf2_r5  0.0883    0.4155    0.3272  0.0489  0.1219     20.0
gutenberg       gemini-001 fixed_lodo    eig2  0.0770    0.2301    0.1531  0.0395  0.1146     20.0
   turing   openai-3-small       main vmf2_r5  0.4472       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small       main vmf2_r3  0.4791       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small       main vmf2_r8  0.4296       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small       main    eig2  0.6947       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small  fixed_all vmf2_r5 -0.1036    0.3443    0.4478 -0.1536 -0.0544      1.0
   turing   openai-3-small  fixed_all    eig2 -0.0013    0.2731    0.2744 -0.0461  0.0388      1.0
   turing   openai-3-small fixed_lodo vmf2_r5 -0.1012    0.3285    0.4297 -0.1287 -0.0730     20.0
   turing   openai-3-small fixed_lodo    eig2  0.0302    0.2982    0.2680 -0.0141  0.0718     20.0
   turing   openai-3-large       main vmf2_r5  0.4779       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large       main vmf2_r3  0.4836       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large       main vmf2_r8  0.4670       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large       main    eig2  0.7044       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large  fixed_all vmf2_r5 -0.0055    0.4176    0.4231 -0.0442  0.0260      1.0
   turing   openai-3-large  fixed_all    eig2  0.0329    0.3312    0.2983 -0.0159  0.0784      1.0
   turing   openai-3-large fixed_lodo vmf2_r5 -0.0264    0.3790    0.4054 -0.0596  0.0066     20.0
   turing   openai-3-large fixed_lodo    eig2  0.0395    0.3280    0.2885  0.0031  0.0786     20.0
   turing       vertex-005       main vmf2_r5  0.4681       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005       main vmf2_r3  0.4678       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005       main vmf2_r8  0.4630       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005       main    eig2  0.6661       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005  fixed_all vmf2_r5 -0.0034    0.3984    0.4017 -0.0638  0.0488      1.0
   turing       vertex-005  fixed_all    eig2  0.0145    0.3304    0.3160 -0.0262  0.0550      1.0
   turing       vertex-005 fixed_lodo vmf2_r5  0.0082    0.4059    0.3977 -0.0353  0.0533     20.0
   turing       vertex-005 fixed_lodo    eig2  0.0311    0.3332    0.3020 -0.0136  0.0779     20.0
   turing modernbert-embed       main vmf2_r5  0.4141       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed       main vmf2_r3  0.4500       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed       main vmf2_r8  0.3722       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed       main    eig2  0.6463       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed  fixed_all vmf2_r5 -0.0396    0.3463    0.3860 -0.0799  0.0190      1.0
   turing modernbert-embed  fixed_all    eig2  0.1224    0.4748    0.3524  0.0917  0.1505      1.0
   turing modernbert-embed fixed_lodo vmf2_r5 -0.0693    0.3207    0.3900 -0.1064 -0.0325     20.0
   turing modernbert-embed fixed_lodo    eig2  0.1148    0.4715    0.3567  0.0894  0.1395     20.0
   turing         bge-base       main vmf2_r5  0.4975       NaN       NaN     NaN     NaN      NaN
   turing         bge-base       main vmf2_r3  0.5088       NaN       NaN     NaN     NaN      NaN
   turing         bge-base       main vmf2_r8  0.4881       NaN       NaN     NaN     NaN      NaN
   turing         bge-base       main    eig2  0.5892       NaN       NaN     NaN     NaN      NaN
   turing         bge-base  fixed_all vmf2_r5  0.0609    0.4079    0.3470  0.0342  0.0877      1.0
   turing         bge-base  fixed_all    eig2  0.0634    0.3926    0.3291  0.0148  0.1024      1.0
   turing         bge-base fixed_lodo vmf2_r5  0.0353    0.3946    0.3593  0.0098  0.0615     20.0
   turing         bge-base fixed_lodo    eig2  0.0840    0.4082    0.3243  0.0544  0.1109     20.0
   turing       gemini-001       main vmf2_r5  0.3879       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001       main vmf2_r3  0.4246       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001       main vmf2_r8  0.3604       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001       main    eig2  0.2962       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001  fixed_all vmf2_r5  0.0322    0.3394    0.3072 -0.0302  0.0926      1.0
   turing       gemini-001  fixed_all    eig2  0.1049    0.1677    0.0628  0.0688  0.1369      1.0
   turing       gemini-001 fixed_lodo vmf2_r5  0.0194    0.3355    0.3162 -0.0376  0.0728     20.0
   turing       gemini-001 fixed_lodo    eig2  0.1162    0.1725    0.0563  0.0828  0.1466     20.0
   fields   openai-3-small       main vmf2_r5  0.4362       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small       main vmf2_r3  0.4658       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small       main vmf2_r8  0.4239       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small       main    eig2  0.6702       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small  fixed_all vmf2_r5  0.0164    0.2923    0.2759 -0.0155  0.0469      1.0
   fields   openai-3-small  fixed_all    eig2  0.0403    0.2243    0.1840 -0.0075  0.0778      1.0
   fields   openai-3-small fixed_lodo vmf2_r5  0.0177    0.2931    0.2754 -0.0190  0.0548     20.0
   fields   openai-3-small fixed_lodo    eig2  0.0444    0.2278    0.1834  0.0071  0.0757     20.0
   fields   openai-3-large       main vmf2_r5  0.4899       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large       main vmf2_r3  0.5091       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large       main vmf2_r8  0.4579       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large       main    eig2  0.5459       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large  fixed_all vmf2_r5 -0.0928    0.3504    0.4433 -0.1328 -0.0529      1.0
   fields   openai-3-large  fixed_all    eig2 -0.0612    0.1504    0.2116 -0.0896 -0.0293      1.0
   fields   openai-3-large fixed_lodo vmf2_r5 -0.1047    0.3457    0.4504 -0.1414 -0.0697     20.0
   fields   openai-3-large fixed_lodo    eig2 -0.0533    0.1591    0.2124 -0.0864 -0.0177     20.0
   fields       vertex-005       main vmf2_r5  0.5939       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005       main vmf2_r3  0.5883       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005       main vmf2_r8  0.6008       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005       main    eig2  0.6785       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005  fixed_all vmf2_r5  0.0487    0.3818    0.3331 -0.0126  0.1037      1.0
   fields       vertex-005  fixed_all    eig2 -0.0641    0.2092    0.2734 -0.1245  0.0021      1.0
   fields       vertex-005 fixed_lodo vmf2_r5  0.0295    0.3594    0.3299 -0.0329  0.0998     20.0
   fields       vertex-005 fixed_lodo    eig2 -0.0662    0.2093    0.2755 -0.1258  0.0034     20.0
   fields modernbert-embed       main vmf2_r5  0.3651       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed       main vmf2_r3  0.4470       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed       main vmf2_r8  0.3015       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed       main    eig2  0.5191       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed  fixed_all vmf2_r5  0.0051    0.2696    0.2646 -0.0403  0.0484      1.0
   fields modernbert-embed  fixed_all    eig2  0.1070    0.3996    0.2927  0.0738  0.1396      1.0
   fields modernbert-embed fixed_lodo vmf2_r5 -0.0119    0.2597    0.2716 -0.0505  0.0314     20.0
   fields modernbert-embed fixed_lodo    eig2  0.1160    0.4037    0.2877  0.0877  0.1474     20.0
   fields         bge-base       main vmf2_r5  0.5356       NaN       NaN     NaN     NaN      NaN
   fields         bge-base       main vmf2_r3  0.5753       NaN       NaN     NaN     NaN      NaN
   fields         bge-base       main vmf2_r8  0.5024       NaN       NaN     NaN     NaN      NaN
   fields         bge-base       main    eig2  0.5596       NaN       NaN     NaN     NaN      NaN
   fields         bge-base  fixed_all vmf2_r5  0.0928    0.4141    0.3212  0.0623  0.1352      1.0
   fields         bge-base  fixed_all    eig2  0.0520    0.3287    0.2767  0.0132  0.0819      1.0
   fields         bge-base fixed_lodo vmf2_r5  0.0528    0.3936    0.3408  0.0203  0.0908     20.0
   fields         bge-base fixed_lodo    eig2  0.0255    0.3166    0.2911 -0.0169  0.0625     20.0
   fields       gemini-001       main vmf2_r5  0.4664       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001       main vmf2_r3  0.5196       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001       main vmf2_r8  0.3942       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001       main    eig2  0.0955       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001  fixed_all vmf2_r5 -0.0162    0.2853    0.3016 -0.0630  0.0337      1.0
   fields       gemini-001  fixed_all    eig2  0.0065    0.0147    0.0082 -0.0007  0.0143      1.0
   fields       gemini-001 fixed_lodo vmf2_r5 -0.0404    0.2645    0.3049 -0.0855  0.0055     20.0
   fields       gemini-001 fixed_lodo    eig2  0.0119    0.0188    0.0068  0.0040  0.0215     20.0

## Longest-edge audit (share of neighbourhoods)

   corpus            model     n   one  two_three  four_plus
    nobel   openai-3-small  3792 0.701      0.183      0.116
    nobel   openai-3-large  3792 0.624      0.208      0.167
    nobel       vertex-005  3792 0.739      0.146      0.114
    nobel modernbert-embed  3792 0.943      0.044      0.013
    nobel         bge-base  3792 0.827      0.116      0.057
    nobel       gemini-001  3792 0.647      0.208      0.146
     wiki   openai-3-small  5398 0.449      0.157      0.393
     wiki   openai-3-large  5398 0.408      0.163      0.429
     wiki       vertex-005  5398 0.368      0.175      0.457
     wiki modernbert-embed  5398 0.783      0.132      0.085
     wiki         bge-base  5398 0.619      0.152      0.229
     wiki       gemini-001  5398 0.529      0.171      0.300
gutenberg   openai-3-small 10278 0.674      0.173      0.153
gutenberg   openai-3-large 10278 0.624      0.194      0.182
gutenberg       vertex-005 10278 0.667      0.185      0.148
gutenberg modernbert-embed 10278 0.896      0.072      0.032
gutenberg         bge-base 10278 0.862      0.093      0.045
gutenberg       gemini-001 10278 0.734      0.151      0.115
   turing   openai-3-small  4107 0.813      0.140      0.047
   turing   openai-3-large  4107 0.751      0.187      0.061
   turing       vertex-005  4107 0.823      0.135      0.042
   turing modernbert-embed  4107 0.959      0.038      0.002
   turing         bge-base  4107 0.892      0.093      0.014
   turing       gemini-001  4107 0.837      0.121      0.042
   fields   openai-3-small  6188 0.844      0.124      0.032
   fields   openai-3-large  6188 0.730      0.184      0.086
   fields       vertex-005  6188 0.862      0.117      0.021
   fields modernbert-embed  6188 0.964      0.035      0.001
   fields         bge-base  6188 0.888      0.092      0.020
   fields       gemini-001  6188 0.747      0.177      0.076
