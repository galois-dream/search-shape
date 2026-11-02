# vMF statistic (docs/vmf_plan.md), K = 50

   corpus            model   analysis feature     est  auc_real  auc_ctrl      lo      hi  n_draws
    nobel   openai-3-small       main vmf2_r5  0.3636       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small       main vmf2_r3  0.4183       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small       main vmf2_r8  0.3412       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small       main    eig2  0.6065       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-small  fixed_all vmf2_r5  0.0982    0.3886    0.2904  0.0500  0.1418      1.0
    nobel   openai-3-small  fixed_all    eig2  0.0690    0.3635    0.2945  0.0233  0.1116      1.0
    nobel   openai-3-small fixed_lodo vmf2_r5  0.1077    0.3956    0.2879  0.0799  0.1403     20.0
    nobel   openai-3-small fixed_lodo    eig2  0.0960    0.3912    0.2952  0.0569  0.1269     20.0
    nobel   openai-3-large       main vmf2_r5  0.3923       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large       main vmf2_r3  0.4214       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large       main vmf2_r8  0.3729       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large       main    eig2  0.6310       NaN       NaN     NaN     NaN      NaN
    nobel   openai-3-large  fixed_all vmf2_r5  0.0323    0.3785    0.3462 -0.0267  0.0760      1.0
    nobel   openai-3-large  fixed_all    eig2  0.0136    0.3569    0.3433 -0.0280  0.0472      1.0
    nobel   openai-3-large fixed_lodo vmf2_r5  0.0369    0.3824    0.3456  0.0032  0.0672     20.0
    nobel   openai-3-large fixed_lodo    eig2  0.0283    0.3711    0.3428 -0.0078  0.0637     20.0
    nobel       vertex-005       main vmf2_r5  0.4465       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005       main vmf2_r3  0.4772       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005       main vmf2_r8  0.4304       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005       main    eig2  0.6049       NaN       NaN     NaN     NaN      NaN
    nobel       vertex-005  fixed_all vmf2_r5 -0.1075    0.4015    0.5090 -0.1600 -0.0628      1.0
    nobel       vertex-005  fixed_all    eig2 -0.0999    0.2535    0.3534 -0.1482 -0.0510      1.0
    nobel       vertex-005 fixed_lodo vmf2_r5 -0.0993    0.3955    0.4948 -0.1355 -0.0683     20.0
    nobel       vertex-005 fixed_lodo    eig2 -0.1153    0.2442    0.3595 -0.1526 -0.0747     20.0
    nobel modernbert-embed       main vmf2_r5  0.4918       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed       main vmf2_r3  0.4976       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed       main vmf2_r8  0.4819       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed       main    eig2  0.7218       NaN       NaN     NaN     NaN      NaN
    nobel modernbert-embed  fixed_all vmf2_r5  0.0719    0.4445    0.3726  0.0401  0.1034      1.0
    nobel modernbert-embed  fixed_all    eig2  0.1403    0.5857    0.4454  0.0966  0.1819      1.0
    nobel modernbert-embed fixed_lodo vmf2_r5  0.1019    0.4635    0.3616  0.0643  0.1441     20.0
    nobel modernbert-embed fixed_lodo    eig2  0.1521    0.5959    0.4438  0.1140  0.1929     20.0
    nobel         bge-base       main vmf2_r5  0.4383       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base       main vmf2_r3  0.4401       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base       main vmf2_r8  0.4396       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base       main    eig2  0.6246       NaN       NaN     NaN     NaN      NaN
    nobel         bge-base  fixed_all vmf2_r5  0.1518    0.4512    0.2994  0.1093  0.1902      1.0
    nobel         bge-base  fixed_all    eig2  0.0398    0.4244    0.3846  0.0033  0.0723      1.0
    nobel         bge-base fixed_lodo vmf2_r5  0.1503    0.4503    0.3000  0.1220  0.1718     20.0
    nobel         bge-base fixed_lodo    eig2  0.0567    0.4368    0.3800  0.0217  0.0906     20.0
    nobel       gemini-001       main vmf2_r5  0.3415       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001       main vmf2_r3  0.3753       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001       main vmf2_r8  0.3145       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001       main    eig2  0.2567       NaN       NaN     NaN     NaN      NaN
    nobel       gemini-001  fixed_all vmf2_r5  0.0662    0.3269    0.2607  0.0054  0.1106      1.0
    nobel       gemini-001  fixed_all    eig2  0.1299    0.1710    0.0410  0.0853  0.1697      1.0
    nobel       gemini-001 fixed_lodo vmf2_r5  0.0840    0.3254    0.2414  0.0389  0.1282     20.0
    nobel       gemini-001 fixed_lodo    eig2  0.1155    0.1547    0.0392  0.0750  0.1578     20.0
    nobel   openai-ada-002       main vmf2_r5  0.3621       NaN       NaN     NaN     NaN      NaN
    nobel   openai-ada-002       main vmf2_r3  0.3994       NaN       NaN     NaN     NaN      NaN
    nobel   openai-ada-002       main vmf2_r8  0.3468       NaN       NaN     NaN     NaN      NaN
    nobel   openai-ada-002       main    eig2  0.6146       NaN       NaN     NaN     NaN      NaN
    nobel   openai-ada-002  fixed_all vmf2_r5  0.0702    0.3922    0.3221  0.0255  0.1100      1.0
    nobel   openai-ada-002  fixed_all    eig2  0.0698    0.3734    0.3036  0.0341  0.1029      1.0
    nobel   openai-ada-002 fixed_lodo vmf2_r5  0.1257    0.4319    0.3062  0.0900  0.1659     20.0
    nobel   openai-ada-002 fixed_lodo    eig2  0.0966    0.4061    0.3096  0.0489  0.1372     20.0
    nobel           bge-m3       main vmf2_r5  0.4606       NaN       NaN     NaN     NaN      NaN
    nobel           bge-m3       main vmf2_r3  0.4741       NaN       NaN     NaN     NaN      NaN
    nobel           bge-m3       main vmf2_r8  0.4620       NaN       NaN     NaN     NaN      NaN
    nobel           bge-m3       main    eig2  0.6832       NaN       NaN     NaN     NaN      NaN
    nobel           bge-m3  fixed_all vmf2_r5 -0.0486    0.3767    0.4253 -0.1068  0.0089      1.0
    nobel           bge-m3  fixed_all    eig2 -0.0844    0.3956    0.4800 -0.1287 -0.0418      1.0
    nobel           bge-m3 fixed_lodo vmf2_r5 -0.0201    0.4077    0.4277 -0.0674  0.0248     20.0
    nobel           bge-m3 fixed_lodo    eig2 -0.0248    0.4504    0.4751 -0.0791  0.0433     20.0
    nobel       qwen3-0.6b       main vmf2_r5  0.4141       NaN       NaN     NaN     NaN      NaN
    nobel       qwen3-0.6b       main vmf2_r3  0.4324       NaN       NaN     NaN     NaN      NaN
    nobel       qwen3-0.6b       main vmf2_r8  0.4109       NaN       NaN     NaN     NaN      NaN
    nobel       qwen3-0.6b       main    eig2  0.5273       NaN       NaN     NaN     NaN      NaN
    nobel       qwen3-0.6b  fixed_all vmf2_r5  0.0523    0.3973    0.3451  0.0140  0.0848      1.0
    nobel       qwen3-0.6b  fixed_all    eig2  0.0614    0.2861    0.2247  0.0295  0.0898      1.0
    nobel       qwen3-0.6b fixed_lodo vmf2_r5  0.0345    0.4199    0.3854  0.0076  0.0599     20.0
    nobel       qwen3-0.6b fixed_lodo    eig2  0.0529    0.3017    0.2488  0.0095  0.1013     20.0
     wiki   openai-3-small       main vmf2_r5  0.4745       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small       main vmf2_r3  0.5126       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small       main vmf2_r8  0.4759       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small       main    eig2  0.4828       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-small  fixed_all vmf2_r5  0.0720    0.4697    0.3977  0.0246  0.1141      1.0
     wiki   openai-3-small  fixed_all    eig2 -0.0400    0.1986    0.2385 -0.0820 -0.0025      1.0
     wiki   openai-3-small fixed_lodo vmf2_r5  0.1399    0.4695    0.3296  0.1119  0.1679     20.0
     wiki   openai-3-small fixed_lodo    eig2 -0.0075    0.2281    0.2356 -0.0448  0.0286     20.0
     wiki   openai-3-large       main vmf2_r5  0.4331       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large       main vmf2_r3  0.4828       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large       main vmf2_r8  0.4913       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large       main    eig2  0.5743       NaN       NaN     NaN     NaN      NaN
     wiki   openai-3-large  fixed_all vmf2_r5  0.0468    0.4822    0.4354  0.0095  0.0789      1.0
     wiki   openai-3-large  fixed_all    eig2 -0.0620    0.2269    0.2889 -0.1085 -0.0295      1.0
     wiki   openai-3-large fixed_lodo vmf2_r5  0.0714    0.4534    0.3820  0.0421  0.0997     20.0
     wiki   openai-3-large fixed_lodo    eig2 -0.0483    0.2201    0.2684 -0.0871 -0.0112     20.0
     wiki       vertex-005       main vmf2_r5  0.7514       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005       main vmf2_r3  0.7274       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005       main vmf2_r8  0.7528       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005       main    eig2  0.6034       NaN       NaN     NaN     NaN      NaN
     wiki       vertex-005  fixed_all vmf2_r5  0.0760    0.5407    0.4648  0.0372  0.1187      1.0
     wiki       vertex-005  fixed_all    eig2 -0.0069    0.3585    0.3654 -0.0401  0.0223      1.0
     wiki       vertex-005 fixed_lodo vmf2_r5  0.0933    0.5493    0.4560  0.0681  0.1219     20.0
     wiki       vertex-005 fixed_lodo    eig2 -0.0092    0.3575    0.3667 -0.0380  0.0151     20.0
     wiki modernbert-embed       main vmf2_r5  0.5660       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed       main vmf2_r3  0.5663       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed       main vmf2_r8  0.5719       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed       main    eig2  0.5750       NaN       NaN     NaN     NaN      NaN
     wiki modernbert-embed  fixed_all vmf2_r5 -0.0020    0.3878    0.3898 -0.0489  0.0453      1.0
     wiki modernbert-embed  fixed_all    eig2 -0.0014    0.3099    0.3114 -0.0377  0.0315      1.0
     wiki modernbert-embed fixed_lodo vmf2_r5  0.0227    0.3799    0.3572 -0.0072  0.0536     20.0
     wiki modernbert-embed fixed_lodo    eig2 -0.0218    0.2903    0.3120 -0.0535  0.0063     20.0
     wiki         bge-base       main vmf2_r5  0.4882       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base       main vmf2_r3  0.5139       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base       main vmf2_r8  0.4858       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base       main    eig2  0.4720       NaN       NaN     NaN     NaN      NaN
     wiki         bge-base  fixed_all vmf2_r5 -0.0059    0.4839    0.4899 -0.0383  0.0310      1.0
     wiki         bge-base  fixed_all    eig2 -0.0246    0.2824    0.3071 -0.0660  0.0066      1.0
     wiki         bge-base fixed_lodo vmf2_r5  0.0089    0.4931    0.4842 -0.0205  0.0390     20.0
     wiki         bge-base fixed_lodo    eig2 -0.0255    0.2874    0.3130 -0.0598  0.0053     20.0
     wiki       gemini-001       main vmf2_r5  0.2908       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001       main vmf2_r3  0.3504       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001       main vmf2_r8  0.2970       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001       main    eig2  0.1857       NaN       NaN     NaN     NaN      NaN
     wiki       gemini-001  fixed_all vmf2_r5  0.0621    0.3883    0.3262  0.0075  0.0996      1.0
     wiki       gemini-001  fixed_all    eig2  0.0241    0.1177    0.0936 -0.0185  0.0676      1.0
     wiki       gemini-001 fixed_lodo vmf2_r5  0.1253    0.4618    0.3365  0.0995  0.1533     20.0
     wiki       gemini-001 fixed_lodo    eig2  0.0376    0.1382    0.1006  0.0064  0.0741     20.0
     wiki   openai-ada-002       main vmf2_r5  0.4327       NaN       NaN     NaN     NaN      NaN
     wiki   openai-ada-002       main vmf2_r3  0.4794       NaN       NaN     NaN     NaN      NaN
     wiki   openai-ada-002       main vmf2_r8  0.4531       NaN       NaN     NaN     NaN      NaN
     wiki   openai-ada-002       main    eig2  0.4876       NaN       NaN     NaN     NaN      NaN
     wiki   openai-ada-002  fixed_all vmf2_r5  0.0958    0.4489    0.3531  0.0571  0.1257      1.0
     wiki   openai-ada-002  fixed_all    eig2 -0.0320    0.2570    0.2890 -0.0775  0.0064      1.0
     wiki   openai-ada-002 fixed_lodo vmf2_r5  0.0924    0.4499    0.3575  0.0568  0.1249     20.0
     wiki   openai-ada-002 fixed_lodo    eig2 -0.0055    0.2793    0.2847 -0.0382  0.0209     20.0
     wiki           bge-m3       main vmf2_r5  0.4933       NaN       NaN     NaN     NaN      NaN
     wiki           bge-m3       main vmf2_r3  0.5186       NaN       NaN     NaN     NaN      NaN
     wiki           bge-m3       main vmf2_r8  0.4890       NaN       NaN     NaN     NaN      NaN
     wiki           bge-m3       main    eig2  0.4666       NaN       NaN     NaN     NaN      NaN
     wiki           bge-m3  fixed_all vmf2_r5  0.0496    0.4450    0.3953  0.0170  0.0783      1.0
     wiki           bge-m3  fixed_all    eig2 -0.0616    0.2610    0.3226 -0.1092 -0.0254      1.0
     wiki           bge-m3 fixed_lodo vmf2_r5  0.1125    0.4801    0.3676  0.0863  0.1429     20.0
     wiki           bge-m3 fixed_lodo    eig2 -0.0293    0.2873    0.3166 -0.0580 -0.0020     20.0
     wiki       qwen3-0.6b       main vmf2_r5  0.4134       NaN       NaN     NaN     NaN      NaN
     wiki       qwen3-0.6b       main vmf2_r3  0.4648       NaN       NaN     NaN     NaN      NaN
     wiki       qwen3-0.6b       main vmf2_r8  0.4381       NaN       NaN     NaN     NaN      NaN
     wiki       qwen3-0.6b       main    eig2  0.4866       NaN       NaN     NaN     NaN      NaN
     wiki       qwen3-0.6b  fixed_all vmf2_r5 -0.0267    0.4176    0.4443 -0.0671  0.0078      1.0
     wiki       qwen3-0.6b  fixed_all    eig2 -0.0747    0.2597    0.3344 -0.1144 -0.0430      1.0
     wiki       qwen3-0.6b fixed_lodo vmf2_r5  0.0536    0.4794    0.4258  0.0243  0.0869     20.0
     wiki       qwen3-0.6b fixed_lodo    eig2 -0.0466    0.2754    0.3219 -0.0805 -0.0174     20.0
gutenberg   openai-3-small       main vmf2_r5  0.2658       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small       main vmf2_r3  0.2768       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small       main vmf2_r8  0.2366       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small       main    eig2  0.3488       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-small  fixed_all vmf2_r5 -0.0597    0.3338    0.3935 -0.0950 -0.0285      1.0
gutenberg   openai-3-small  fixed_all    eig2 -0.0056    0.2982    0.3039 -0.0398  0.0182      1.0
gutenberg   openai-3-small fixed_lodo vmf2_r5 -0.0356    0.3511    0.3867 -0.0632 -0.0123     20.0
gutenberg   openai-3-small fixed_lodo    eig2 -0.0173    0.3068    0.3241 -0.0485  0.0086     20.0
gutenberg   openai-3-large       main vmf2_r5  0.2429       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large       main vmf2_r3  0.2469       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large       main vmf2_r8  0.2213       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large       main    eig2  0.3097       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-3-large  fixed_all vmf2_r5 -0.0659    0.3300    0.3959 -0.1047 -0.0349      1.0
gutenberg   openai-3-large  fixed_all    eig2 -0.0167    0.2958    0.3125 -0.0513  0.0085      1.0
gutenberg   openai-3-large fixed_lodo vmf2_r5 -0.0376    0.3564    0.3940 -0.0627 -0.0147     20.0
gutenberg   openai-3-large fixed_lodo    eig2 -0.0293    0.3055    0.3348 -0.0645  0.0024     20.0
gutenberg       vertex-005       main vmf2_r5  0.3156       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005       main vmf2_r3  0.3249       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005       main vmf2_r8  0.2982       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005       main    eig2  0.3225       NaN       NaN     NaN     NaN      NaN
gutenberg       vertex-005  fixed_all vmf2_r5 -0.0275    0.4144    0.4419 -0.0612  0.0029      1.0
gutenberg       vertex-005  fixed_all    eig2 -0.0156    0.3152    0.3308 -0.0517  0.0128      1.0
gutenberg       vertex-005 fixed_lodo vmf2_r5 -0.0494    0.3906    0.4400 -0.0767 -0.0201     20.0
gutenberg       vertex-005 fixed_lodo    eig2 -0.0626    0.2847    0.3474 -0.0902 -0.0385     20.0
gutenberg modernbert-embed       main vmf2_r5  0.3717       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed       main vmf2_r3  0.3869       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed       main vmf2_r8  0.3643       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed       main    eig2  0.4734       NaN       NaN     NaN     NaN      NaN
gutenberg modernbert-embed  fixed_all vmf2_r5 -0.0292    0.4135    0.4428 -0.0579  0.0001      1.0
gutenberg modernbert-embed  fixed_all    eig2 -0.0068    0.4504    0.4572 -0.0316  0.0181      1.0
gutenberg modernbert-embed fixed_lodo vmf2_r5 -0.0273    0.4036    0.4309 -0.0485 -0.0094     20.0
gutenberg modernbert-embed fixed_lodo    eig2 -0.0121    0.4459    0.4581 -0.0257  0.0013     20.0
gutenberg         bge-base       main vmf2_r5  0.3991       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base       main vmf2_r3  0.4187       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base       main vmf2_r8  0.3909       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base       main    eig2  0.5098       NaN       NaN     NaN     NaN      NaN
gutenberg         bge-base  fixed_all vmf2_r5 -0.0179    0.4079    0.4258 -0.0527  0.0171      1.0
gutenberg         bge-base  fixed_all    eig2  0.0313    0.4227    0.3914  0.0014  0.0579      1.0
gutenberg         bge-base fixed_lodo vmf2_r5 -0.0142    0.4158    0.4300 -0.0353  0.0058     20.0
gutenberg         bge-base fixed_lodo    eig2  0.0293    0.4218    0.3925  0.0086  0.0497     20.0
gutenberg       gemini-001       main vmf2_r5  0.2161       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001       main vmf2_r3  0.2338       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001       main vmf2_r8  0.1934       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001       main    eig2  0.1427       NaN       NaN     NaN     NaN      NaN
gutenberg       gemini-001  fixed_all vmf2_r5 -0.0430    0.3057    0.3487 -0.0827 -0.0091      1.0
gutenberg       gemini-001  fixed_all    eig2  0.0345    0.2155    0.1810 -0.0054  0.0687      1.0
gutenberg       gemini-001 fixed_lodo vmf2_r5  0.0356    0.3822    0.3466  0.0075  0.0634     20.0
gutenberg       gemini-001 fixed_lodo    eig2  0.0513    0.2430    0.1917  0.0111  0.0907     20.0
gutenberg   openai-ada-002       main vmf2_r5  0.3012       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-ada-002       main vmf2_r3  0.3106       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-ada-002       main vmf2_r8  0.2803       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-ada-002       main    eig2  0.3934       NaN       NaN     NaN     NaN      NaN
gutenberg   openai-ada-002  fixed_all vmf2_r5 -0.0126    0.3558    0.3684 -0.0555  0.0242      1.0
gutenberg   openai-ada-002  fixed_all    eig2  0.0416    0.3368    0.2953  0.0064  0.0687      1.0
gutenberg   openai-ada-002 fixed_lodo vmf2_r5  0.0125    0.3867    0.3741 -0.0131  0.0366     20.0
gutenberg   openai-ada-002 fixed_lodo    eig2  0.0372    0.3441    0.3068  0.0086  0.0641     20.0
gutenberg           bge-m3       main vmf2_r5  0.3089       NaN       NaN     NaN     NaN      NaN
gutenberg           bge-m3       main vmf2_r3  0.3112       NaN       NaN     NaN     NaN      NaN
gutenberg           bge-m3       main vmf2_r8  0.2971       NaN       NaN     NaN     NaN      NaN
gutenberg           bge-m3       main    eig2  0.4252       NaN       NaN     NaN     NaN      NaN
gutenberg           bge-m3  fixed_all vmf2_r5 -0.0404    0.3630    0.4035 -0.0737 -0.0124      1.0
gutenberg           bge-m3  fixed_all    eig2 -0.0206    0.3678    0.3884 -0.0537  0.0031      1.0
gutenberg           bge-m3 fixed_lodo vmf2_r5 -0.0148    0.3854    0.4002 -0.0388  0.0066     20.0
gutenberg           bge-m3 fixed_lodo    eig2 -0.0097    0.3821    0.3918 -0.0356  0.0108     20.0
gutenberg       qwen3-0.6b       main vmf2_r5  0.3162       NaN       NaN     NaN     NaN      NaN
gutenberg       qwen3-0.6b       main vmf2_r3  0.3182       NaN       NaN     NaN     NaN      NaN
gutenberg       qwen3-0.6b       main vmf2_r8  0.3072       NaN       NaN     NaN     NaN      NaN
gutenberg       qwen3-0.6b       main    eig2  0.4512       NaN       NaN     NaN     NaN      NaN
gutenberg       qwen3-0.6b  fixed_all vmf2_r5 -0.1097    0.3373    0.4470 -0.1513 -0.0740      1.0
gutenberg       qwen3-0.6b  fixed_all    eig2 -0.0545    0.3412    0.3956 -0.0940 -0.0201      1.0
gutenberg       qwen3-0.6b fixed_lodo vmf2_r5 -0.0889    0.3582    0.4471 -0.1137 -0.0679     20.0
gutenberg       qwen3-0.6b fixed_lodo    eig2 -0.0401    0.3597    0.3998 -0.0711 -0.0155     20.0
   turing   openai-3-small       main vmf2_r5  0.4220       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small       main vmf2_r3  0.4480       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small       main vmf2_r8  0.3926       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small       main    eig2  0.6698       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-small  fixed_all vmf2_r5 -0.0624    0.3253    0.3878 -0.1146 -0.0213      1.0
   turing   openai-3-small  fixed_all    eig2 -0.0026    0.3679    0.3705 -0.0512  0.0383      1.0
   turing   openai-3-small fixed_lodo vmf2_r5 -0.0887    0.2905    0.3792 -0.1143 -0.0624     20.0
   turing   openai-3-small fixed_lodo    eig2 -0.0229    0.3446    0.3674 -0.0560  0.0116     20.0
   turing   openai-3-large       main vmf2_r5  0.4206       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large       main vmf2_r3  0.4340       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large       main vmf2_r8  0.3978       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large       main    eig2  0.6605       NaN       NaN     NaN     NaN      NaN
   turing   openai-3-large  fixed_all vmf2_r5  0.0029    0.4077    0.4048 -0.0346  0.0345      1.0
   turing   openai-3-large  fixed_all    eig2  0.0425    0.4301    0.3876  0.0017  0.0833      1.0
   turing   openai-3-large fixed_lodo vmf2_r5 -0.0394    0.3779    0.4173 -0.0630 -0.0153     20.0
   turing   openai-3-large fixed_lodo    eig2  0.0168    0.4061    0.3893 -0.0213  0.0558     20.0
   turing       vertex-005       main vmf2_r5  0.4526       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005       main vmf2_r3  0.4274       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005       main vmf2_r8  0.4439       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005       main    eig2  0.6258       NaN       NaN     NaN     NaN      NaN
   turing       vertex-005  fixed_all vmf2_r5 -0.0455    0.3872    0.4327 -0.0842 -0.0132      1.0
   turing       vertex-005  fixed_all    eig2 -0.0587    0.3442    0.4029 -0.1021 -0.0216      1.0
   turing       vertex-005 fixed_lodo vmf2_r5 -0.0696    0.3733    0.4429 -0.0986 -0.0405     20.0
   turing       vertex-005 fixed_lodo    eig2 -0.0673    0.3250    0.3923 -0.0991 -0.0367     20.0
   turing modernbert-embed       main vmf2_r5  0.3823       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed       main vmf2_r3  0.4282       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed       main vmf2_r8  0.3520       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed       main    eig2  0.6332       NaN       NaN     NaN     NaN      NaN
   turing modernbert-embed  fixed_all vmf2_r5  0.0040    0.3767    0.3728 -0.0360  0.0505      1.0
   turing modernbert-embed  fixed_all    eig2  0.0921    0.5231    0.4310  0.0683  0.1161      1.0
   turing modernbert-embed fixed_lodo vmf2_r5 -0.0346    0.3593    0.3939 -0.0642  0.0019     20.0
   turing modernbert-embed fixed_lodo    eig2  0.0824    0.5229    0.4405  0.0611  0.1073     20.0
   turing         bge-base       main vmf2_r5  0.4562       NaN       NaN     NaN     NaN      NaN
   turing         bge-base       main vmf2_r3  0.4493       NaN       NaN     NaN     NaN      NaN
   turing         bge-base       main vmf2_r8  0.4425       NaN       NaN     NaN     NaN      NaN
   turing         bge-base       main    eig2  0.5763       NaN       NaN     NaN     NaN      NaN
   turing         bge-base  fixed_all vmf2_r5  0.0049    0.4158    0.4110 -0.0402  0.0391      1.0
   turing         bge-base  fixed_all    eig2  0.0085    0.4211    0.4126 -0.0246  0.0405      1.0
   turing         bge-base fixed_lodo vmf2_r5 -0.0209    0.3745    0.3954 -0.0444  0.0011     20.0
   turing         bge-base fixed_lodo    eig2  0.0068    0.4132    0.4064 -0.0245  0.0341     20.0
   turing       gemini-001       main vmf2_r5  0.3157       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001       main vmf2_r3  0.3515       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001       main vmf2_r8  0.2764       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001       main    eig2  0.2482       NaN       NaN     NaN     NaN      NaN
   turing       gemini-001  fixed_all vmf2_r5 -0.0333    0.3130    0.3463 -0.0953  0.0245      1.0
   turing       gemini-001  fixed_all    eig2  0.0796    0.2356    0.1560  0.0203  0.1280      1.0
   turing       gemini-001 fixed_lodo vmf2_r5 -0.0380    0.2968    0.3348 -0.0769  0.0004     20.0
   turing       gemini-001 fixed_lodo    eig2  0.0707    0.2298    0.1591  0.0360  0.1031     20.0
   turing   openai-ada-002       main vmf2_r5  0.4502       NaN       NaN     NaN     NaN      NaN
   turing   openai-ada-002       main vmf2_r3  0.4434       NaN       NaN     NaN     NaN      NaN
   turing   openai-ada-002       main vmf2_r8  0.4313       NaN       NaN     NaN     NaN      NaN
   turing   openai-ada-002       main    eig2  0.6342       NaN       NaN     NaN     NaN      NaN
   turing   openai-ada-002  fixed_all vmf2_r5 -0.1022    0.3759    0.4781 -0.1454 -0.0696      1.0
   turing   openai-ada-002  fixed_all    eig2 -0.0068    0.3777    0.3845 -0.0494  0.0235      1.0
   turing   openai-ada-002 fixed_lodo vmf2_r5 -0.1186    0.3550    0.4736 -0.1608 -0.0774     20.0
   turing   openai-ada-002 fixed_lodo    eig2 -0.0102    0.3721    0.3823 -0.0468  0.0231     20.0
   turing           bge-m3       main vmf2_r5  0.4274       NaN       NaN     NaN     NaN      NaN
   turing           bge-m3       main vmf2_r3  0.4427       NaN       NaN     NaN     NaN      NaN
   turing           bge-m3       main vmf2_r8  0.4194       NaN       NaN     NaN     NaN      NaN
   turing           bge-m3       main    eig2  0.6515       NaN       NaN     NaN     NaN      NaN
   turing           bge-m3  fixed_all vmf2_r5 -0.1039    0.3636    0.4675 -0.1588 -0.0615      1.0
   turing           bge-m3  fixed_all    eig2  0.0139    0.4101    0.3962 -0.0319  0.0510      1.0
   turing           bge-m3 fixed_lodo vmf2_r5 -0.1165    0.3584    0.4749 -0.1484 -0.0875     20.0
   turing           bge-m3 fixed_lodo    eig2  0.0418    0.4441    0.4023  0.0087  0.0762     20.0
   turing       qwen3-0.6b       main vmf2_r5  0.5006       NaN       NaN     NaN     NaN      NaN
   turing       qwen3-0.6b       main vmf2_r3  0.4836       NaN       NaN     NaN     NaN      NaN
   turing       qwen3-0.6b       main vmf2_r8  0.4957       NaN       NaN     NaN     NaN      NaN
   turing       qwen3-0.6b       main    eig2  0.5504       NaN       NaN     NaN     NaN      NaN
   turing       qwen3-0.6b  fixed_all vmf2_r5 -0.0345    0.4032    0.4377 -0.0722 -0.0056      1.0
   turing       qwen3-0.6b  fixed_all    eig2  0.0573    0.3209    0.2636  0.0174  0.0885      1.0
   turing       qwen3-0.6b fixed_lodo vmf2_r5 -0.0370    0.4025    0.4395 -0.0658 -0.0109     20.0
   turing       qwen3-0.6b fixed_lodo    eig2  0.0506    0.3111    0.2605  0.0140  0.0887     20.0
   fields   openai-3-small       main vmf2_r5  0.3184       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small       main vmf2_r3  0.3326       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small       main vmf2_r8  0.3014       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small       main    eig2  0.5837       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-small  fixed_all vmf2_r5 -0.0200    0.2583    0.2783 -0.0589  0.0135      1.0
   fields   openai-3-small  fixed_all    eig2 -0.0306    0.2515    0.2821 -0.0804  0.0101      1.0
   fields   openai-3-small fixed_lodo vmf2_r5 -0.0223    0.2671    0.2894 -0.0531  0.0118     20.0
   fields   openai-3-small fixed_lodo    eig2 -0.0214    0.2652    0.2866 -0.0586  0.0134     20.0
   fields   openai-3-large       main vmf2_r5  0.3226       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large       main vmf2_r3  0.3578       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large       main vmf2_r8  0.2959       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large       main    eig2  0.4372       NaN       NaN     NaN     NaN      NaN
   fields   openai-3-large  fixed_all vmf2_r5 -0.1099    0.3153    0.4252 -0.1470 -0.0773      1.0
   fields   openai-3-large  fixed_all    eig2 -0.0782    0.1989    0.2771 -0.1199 -0.0419      1.0
   fields   openai-3-large fixed_lodo vmf2_r5 -0.1139    0.3149    0.4288 -0.1438 -0.0855     20.0
   fields   openai-3-large fixed_lodo    eig2 -0.0961    0.1893    0.2854 -0.1253 -0.0658     20.0
   fields       vertex-005       main vmf2_r5  0.5073       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005       main vmf2_r3  0.4903       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005       main vmf2_r8  0.5082       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005       main    eig2  0.6153       NaN       NaN     NaN     NaN      NaN
   fields       vertex-005  fixed_all vmf2_r5  0.0397    0.4059    0.3662 -0.0040  0.0807      1.0
   fields       vertex-005  fixed_all    eig2 -0.1102    0.2177    0.3279 -0.1591 -0.0627      1.0
   fields       vertex-005 fixed_lodo vmf2_r5  0.0591    0.4195    0.3604  0.0192  0.1041     20.0
   fields       vertex-005 fixed_lodo    eig2 -0.1319    0.1975    0.3294 -0.1843 -0.0677     20.0
   fields modernbert-embed       main vmf2_r5  0.3815       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed       main vmf2_r3  0.4460       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed       main vmf2_r8  0.3315       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed       main    eig2  0.6358       NaN       NaN     NaN     NaN      NaN
   fields modernbert-embed  fixed_all vmf2_r5  0.0099    0.3233    0.3134 -0.0313  0.0514      1.0
   fields modernbert-embed  fixed_all    eig2  0.1296    0.4864    0.3568  0.0807  0.1675      1.0
   fields modernbert-embed fixed_lodo vmf2_r5  0.0109    0.3220    0.3111 -0.0283  0.0504     20.0
   fields modernbert-embed fixed_lodo    eig2  0.1281    0.4933    0.3652  0.0992  0.1563     20.0
   fields         bge-base       main vmf2_r5  0.4263       NaN       NaN     NaN     NaN      NaN
   fields         bge-base       main vmf2_r3  0.4572       NaN       NaN     NaN     NaN      NaN
   fields         bge-base       main vmf2_r8  0.3959       NaN       NaN     NaN     NaN      NaN
   fields         bge-base       main    eig2  0.4886       NaN       NaN     NaN     NaN      NaN
   fields         bge-base  fixed_all vmf2_r5  0.0518    0.3937    0.3419  0.0254  0.0820      1.0
   fields         bge-base  fixed_all    eig2  0.0525    0.3498    0.2972  0.0173  0.0829      1.0
   fields         bge-base fixed_lodo vmf2_r5  0.0279    0.3780    0.3501 -0.0006  0.0584     20.0
   fields         bge-base fixed_lodo    eig2  0.0301    0.3395    0.3094 -0.0089  0.0634     20.0
   fields       gemini-001       main vmf2_r5  0.2414       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001       main vmf2_r3  0.3196       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001       main vmf2_r8  0.1812       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001       main    eig2  0.0746       NaN       NaN     NaN     NaN      NaN
   fields       gemini-001  fixed_all vmf2_r5 -0.0694    0.2222    0.2916 -0.1077 -0.0317      1.0
   fields       gemini-001  fixed_all    eig2  0.0192    0.0420    0.0228  0.0054  0.0312      1.0
   fields       gemini-001 fixed_lodo vmf2_r5 -0.0645    0.2214    0.2859 -0.0884 -0.0405     20.0
   fields       gemini-001 fixed_lodo    eig2  0.0179    0.0428    0.0249  0.0028  0.0333     20.0
   fields   openai-ada-002       main vmf2_r5  0.4539       NaN       NaN     NaN     NaN      NaN
   fields   openai-ada-002       main vmf2_r3  0.4658       NaN       NaN     NaN     NaN      NaN
   fields   openai-ada-002       main vmf2_r8  0.4544       NaN       NaN     NaN     NaN      NaN
   fields   openai-ada-002       main    eig2  0.6379       NaN       NaN     NaN     NaN      NaN
   fields   openai-ada-002  fixed_all vmf2_r5 -0.0193    0.3709    0.3902 -0.0536  0.0096      1.0
   fields   openai-ada-002  fixed_all    eig2 -0.0247    0.3339    0.3587 -0.0689  0.0074      1.0
   fields   openai-ada-002 fixed_lodo vmf2_r5 -0.0155    0.3654    0.3809 -0.0368  0.0054     20.0
   fields   openai-ada-002 fixed_lodo    eig2 -0.0383    0.3378    0.3761 -0.0679 -0.0131     20.0
   fields           bge-m3       main vmf2_r5  0.4756       NaN       NaN     NaN     NaN      NaN
   fields           bge-m3       main vmf2_r3  0.4821       NaN       NaN     NaN     NaN      NaN
   fields           bge-m3       main vmf2_r8  0.4694       NaN       NaN     NaN     NaN      NaN
   fields           bge-m3       main    eig2  0.6754       NaN       NaN     NaN     NaN      NaN
   fields           bge-m3  fixed_all vmf2_r5  0.0881    0.3218    0.2337  0.0490  0.1233      1.0
   fields           bge-m3  fixed_all    eig2  0.0199    0.2770    0.2571 -0.0271  0.0601      1.0
   fields           bge-m3 fixed_lodo vmf2_r5  0.0440    0.2790    0.2351  0.0213  0.0664     20.0
   fields           bge-m3 fixed_lodo    eig2 -0.0048    0.2562    0.2610 -0.0332  0.0197     20.0
   fields       qwen3-0.6b       main vmf2_r5  0.3453       NaN       NaN     NaN     NaN      NaN
   fields       qwen3-0.6b       main vmf2_r3  0.3610       NaN       NaN     NaN     NaN      NaN
   fields       qwen3-0.6b       main vmf2_r8  0.3411       NaN       NaN     NaN     NaN      NaN
   fields       qwen3-0.6b       main    eig2  0.4724       NaN       NaN     NaN     NaN      NaN
   fields       qwen3-0.6b  fixed_all vmf2_r5 -0.0754    0.2886    0.3640 -0.1105 -0.0409      1.0
   fields       qwen3-0.6b  fixed_all    eig2 -0.0019    0.1930    0.1949 -0.0317  0.0246      1.0
   fields       qwen3-0.6b fixed_lodo vmf2_r5 -0.0543    0.3087    0.3629 -0.0810 -0.0239     20.0
   fields       qwen3-0.6b fixed_lodo    eig2 -0.0201    0.1905    0.2106 -0.0535  0.0137     20.0

## Longest-edge audit (share of neighbourhoods)

   corpus            model     n   one  two_three  four_plus
    nobel   openai-3-small  3792 0.695      0.187      0.118
    nobel   openai-3-large  3792 0.635      0.214      0.151
    nobel       vertex-005  3792 0.752      0.163      0.085
    nobel modernbert-embed  3792 0.926      0.059      0.015
    nobel         bge-base  3792 0.823      0.120      0.057
    nobel       gemini-001  3792 0.659      0.203      0.138
    nobel   openai-ada-002  3792 0.753      0.163      0.084
    nobel           bge-m3  3792 0.774      0.157      0.069
    nobel       qwen3-0.6b  3792 0.727      0.180      0.093
     wiki   openai-3-small  5398 0.425      0.179      0.396
     wiki   openai-3-large  5398 0.405      0.168      0.427
     wiki       vertex-005  5398 0.368      0.177      0.454
     wiki modernbert-embed  5398 0.739      0.160      0.102
     wiki         bge-base  5398 0.614      0.169      0.217
     wiki       gemini-001  5398 0.543      0.173      0.284
     wiki   openai-ada-002  5398 0.504      0.173      0.323
     wiki           bge-m3  5398 0.546      0.185      0.268
     wiki       qwen3-0.6b  5398 0.535      0.174      0.291
gutenberg   openai-3-small 10278 0.665      0.180      0.155
gutenberg   openai-3-large 10278 0.619      0.194      0.187
gutenberg       vertex-005 10278 0.683      0.184      0.133
gutenberg modernbert-embed 10278 0.869      0.094      0.037
gutenberg         bge-base 10278 0.847      0.103      0.050
gutenberg       gemini-001 10278 0.711      0.169      0.120
gutenberg   openai-ada-002 10278 0.752      0.152      0.096
gutenberg           bge-m3 10278 0.782      0.142      0.076
gutenberg       qwen3-0.6b 10278 0.753      0.159      0.088
   turing   openai-3-small  4107 0.785      0.164      0.052
   turing   openai-3-large  4107 0.735      0.203      0.062
   turing       vertex-005  4107 0.813      0.136      0.052
   turing modernbert-embed  4107 0.941      0.053      0.006
   turing         bge-base  4107 0.887      0.097      0.016
   turing       gemini-001  4107 0.812      0.134      0.054
   turing   openai-ada-002  4107 0.807      0.153      0.040
   turing           bge-m3  4107 0.853      0.120      0.028
   turing       qwen3-0.6b  4107 0.828      0.130      0.042
   fields   openai-3-small  6188 0.811      0.145      0.044
   fields   openai-3-large  6188 0.698      0.190      0.112
   fields       vertex-005  6188 0.867      0.108      0.024
   fields modernbert-embed  6188 0.955      0.043      0.002
   fields         bge-base  6188 0.855      0.115      0.031
   fields       gemini-001  6188 0.718      0.186      0.097
   fields   openai-ada-002  6188 0.881      0.102      0.017
   fields           bge-m3  6188 0.883      0.101      0.016
   fields       qwen3-0.6b  6188 0.771      0.160      0.069

## Reproducibility: le_len vs features longbar

   corpus            model  n_matched  max_abs_diff
    nobel   openai-3-small       3792  5.957720e-08
    nobel   openai-3-large       3792  5.959100e-08
    nobel       vertex-005       3792  5.763961e-08
    nobel modernbert-embed       3792  5.955206e-08
    nobel         bge-base       3792  5.959574e-08
    nobel       gemini-001       3792  5.959319e-08
    nobel   openai-ada-002       3792  5.951990e-08
    nobel           bge-m3       3792  5.934550e-08
    nobel       qwen3-0.6b       3792  5.935937e-08
     wiki   openai-3-small       5398  5.959279e-08
     wiki   openai-3-large       5398  5.956924e-08
     wiki       vertex-005       5398  5.694750e-08
     wiki modernbert-embed       5398  5.958848e-08
     wiki         bge-base       5398  5.954642e-08
     wiki       gemini-001       5398  5.959590e-08
     wiki   openai-ada-002       5398  5.954014e-08
     wiki           bge-m3       5398  5.960141e-08
     wiki       qwen3-0.6b       5398  5.955224e-08
gutenberg   openai-3-small      10278  5.949777e-08
gutenberg   openai-3-large      10278  5.958978e-08
gutenberg       vertex-005      10278  3.143922e-08
gutenberg modernbert-embed      10278  5.959123e-08
gutenberg         bge-base      10278  5.959112e-08
gutenberg       gemini-001      10278  5.959489e-08
gutenberg   openai-ada-002      10278  5.958797e-08
gutenberg           bge-m3      10278  5.957206e-08
gutenberg       qwen3-0.6b      10278  5.957690e-08
   turing   openai-3-small       4107  5.953893e-08
   turing   openai-3-large       4107  5.947852e-08
   turing       vertex-005       4107  3.279284e-08
   turing modernbert-embed       4107  5.952138e-08
   turing         bge-base       4107  5.951156e-08
   turing       gemini-001       4107  5.943290e-08
   turing   openai-ada-002       4107  5.954760e-08
   turing           bge-m3       4107  5.948222e-08
   turing       qwen3-0.6b       4107  5.957634e-08
   fields   openai-3-small       6188  5.952090e-08
   fields   openai-3-large       6188  5.955785e-08
   fields       vertex-005       6188  2.979926e-08
   fields modernbert-embed       6188  5.956232e-08
   fields         bge-base       6188  5.956910e-08
   fields       gemini-001       6188  5.954618e-08
   fields   openai-ada-002       6188  5.952844e-08
   fields           bge-m3       6188  5.959502e-08
   fields       qwen3-0.6b       6188  5.958513e-08
