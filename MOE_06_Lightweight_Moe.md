Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical
|     |     |     |     | Vision-Language |     | Models |     |     |     |     |     |
| --- | --- | --- | --- | --------------- | --- | ------ | --- | --- | --- | --- | --- |
SongtaoJiang*1,TuoZheng*1,YanZhang2,YeyingJin2,LiYuan3 andZuozhuLiu1
1ZhejiangUniversity
2NationalUniversityofSingapore
3PekingUniversity
|                 |              | Abstract   |                    |       |          | de Faria      | et al., | 2023; Antol | et       | al., 2015). | Re-     |
| --------------- | ------------ | ---------- | ------------------ | ----- | -------- | ------------- | ------- | ----------- | -------- | ----------- | ------- |
|                 |              |            |                    |       |          | cent progress | on      | Multimodal  | Large    | Language    |         |
| Recent          | advancements |            | in general-purpose |       | or       |               |         |             |          |             |         |
|                 |              |            |                    |       |          | Models        | (MLLMs) | such        | as LLaVA | (Liu        | et al., |
| domain-specific |              | multimodal |                    | large | language |               |         |             |          |             |         |
4202 peS 1  ]VC.sc[  3v73201.4042:viXra models (LLMs) have witnessed remark- 2024a), MiniGPT4-V2 (Chen et al., 2023),
CogVLM(Wangetal.,2023)havedemonstrated
| able progress | for | medical | decision-making. |     |     |     |     |     |     |     |     |
| ------------- | --- | ------- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
However, they are designated for specific greatperformanceacrossmultimodaltasks, how-
classificationorgenerativetasks,andrequire
ever,theyarelesseffectiveinthemedicaldomain
model training or finetuning on large-scale astheyareusuallytrainedwithwebcontentswhich
datasetswithsizeableparametersandtremen-
|                 |     |           |     |       |          | differsignificantlyfromthemedicaldata. |     |     |     | Domain- |     |
| --------------- | --- | --------- | --- | ----- | -------- | -------------------------------------- | --- | --- | --- | ------- | --- |
| dous computing, |     | hindering |     | their | clinical |                                        |     |     |     |         |     |
specificmodelssuchasMed-Flamingo(Mooretal.,
| utility | across | diverse | resource-constrained |     |     |     |     |     |     |     |     |
| ------- | ------ | ------- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
scenarios in practice. In this paper, we 2023), Med-PaLM M (Singhal et al., 2023), and
LLaVA-Med(Lietal.,2024a)exhibitpromisingre-
| propose | a novel | and lightweight |     | framework |     |     |     |     |     |     |     |
| ------- | ------- | --------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
Med-MoE (Mixture-of-Experts) that tackles sultsacrossvariousmedicaltasks,suchasmedical
bothdiscriminativeandgenerativemultimodal visualquestion-answering(Med-VQA),bytraining
medical tasks. The learning of Med-MoE withmedicaldomaindata. However,thesemodels
| consists | of three | steps: | multimodal |     | medical |     |     |     |     |     |     |
| -------- | -------- | ------ | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- |
areusuallytailedforcertainkindsoftasks,suchas
alignment,instructiontuningandrouting,and
close-oropen-endVQA,whileinpractice,medi-
| domain-specific |     | MoE tuning. |     | After | aligning |     |     |     |     |     |     |
| --------------- | --- | ----------- | --- | ----- | -------- | --- | --- | --- | --- | --- | --- |
multimodal medical images with LLM calMLLMsneedtohandlebothdiscriminativeand
generativetaskstoprovidemorereliableandinter-
tokens,wethenenablethemodelfordifferent
multimodal medical tasks with instruction pretabledecisions. Moreover,existingmodelsare
tuning,togetherwithatrainableroutertailored usually obtained with heavy LLMs with sizeable
for expert selection across input modalities. parameters, such as LLama(7B) in LLaVA-Med,
Finally,themodelistunedbyintegratingthe
|             |          |                 |     |     |          | leading | to high | training | and inference | costs | and |
| ----------- | -------- | --------------- | --- | --- | -------- | ------- | ------- | -------- | ------------- | ----- | --- |
| router with | multiple | domain-specific |     |     | experts, |         |         |          |               |       |     |
hinderingitspracticalutilitytobroadclinicalprac-
| which are | selectively |     | activated | and | further |     |     |     |     |     |     |
| --------- | ----------- | --- | --------- | --- | ------- | --- | --- | --- | --- | --- | --- |
titioners.
| empowered   | by  | meta expert. |       | Comprehensive |     |       |           |     |             |     |       |
| ----------- | --- | ------------ | ----- | ------------- | --- | ----- | --------- | --- | ----------- | --- | ----- |
|             |     |              |       |               |     | It is | appealing | but | challenging | to  | build |
| experiments | on  | both         | open- | and close-end |     |       |           |     |             |     |       |
medicalquestionanswering(Med-VQA)and lightweightyeteffectivemedicalMLLMsformul-
image classification tasks across datasets timodaldecision-making (Peterssonetal.,2022;
such as VQA-RAD, SLAKE and Path-VQA Kelly et al., 2019; Liao et al., 2024). Recent re-
demonstrate that our model can achieve searchshowsthatscalingupthequantityorqual-
| performance | superior |     | to or | on  | par with |                    |     |                           |     |     |     |
| ----------- | -------- | --- | ----- | --- | -------- | ------------------ | --- | ------------------------- | --- | --- | --- |
|             |          |     |       |     |          | ityoftrainingdata, |     | aswellasincreasingthesize |     |     |     |
state-of-the-artbaselines,whileonlyrequiring
ofthemodel,canresultinenhancedperformance
| approximately |     | 30%-50% | of activated |     | model |     |     |     |     |     |     |
| ------------- | --- | ------- | ------------ | --- | ----- | --- | --- | --- | --- | --- | --- |
(Gaoetal.,2024;Shietal.,2024;Xueetal.,2024;
| parameters. | Extensiveanalysisandablations |               |     |     |           |                               |     |     |     |                |     |
| ----------- | ----------------------------- | ------------- | --- | --- | --------- | ----------------------------- | --- | --- | --- | -------------- | --- |
|             |                               |               |     |     |           | Shenetal.,2023;Luetal.,2023). |     |     |     | However,train- |     |
| corroborate | the                           | effectiveness |     | and | practical |                               |     |     |     |                |     |
utilityofourmethod. Ourcodeisreleasedat inganddeployingthesemodelsalsodemandsub-
https://github.com/jiangsongtao/Med-MoE. stantial computational resources, rendering them
lessappealingfornumerousclinicalpractitioners
1 Introduction
|     |     |     |     |     |     | who may | lack sufficient |     | computing | power | (Lu |
| --- | --- | --- | --- | --- | --- | ------- | --------------- | --- | --------- | ----- | --- |
Creating systems with human-level multimodal et al., 2023; Crawford, 2021; Thompson et al.,
understanding is essential for medical decision- 2020). Forinstance,manyinstitutionsmaynotpos-
making (Miao et al., 2022; Goyal et al., 2016; sesspowerfulGPUssuchasNVIDIAA100cards

medicaldomainremainsunexplored.
|     |     |     |     |     |     |     | In this | paper, | we propose |     | a lightweight |     | and ef- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ---------- | --- | ------------- | --- | ------- |
fectiveframeworkMed-MoEformultimodalgen-
erativeordiscriminativeMed-VQAandclassifica-
|     |     |     |     |     |     |     | tion tasks. | Our | Med-MoE |     | incorporates |     | multiple |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------- | --- | ------------ | --- | -------- |
domain-specificexpertsalongwithglobalmetaex-
|          |        |                               |     |     |     |     | pert, emulating                  |              | the                          | workflow    | in          | hospitals     | where     |
| -------- | ------ | ----------------------------- | --- | --- | --- | --- | -------------------------------- | ------------ | ---------------------------- | ----------- | ----------- | ------------- | --------- |
|          |        |                               |     |     |     |     | various                          | departments  |                              | collaborate |             | together      | for dis-  |
|          |        |                               |     |     |     |     | easediagnosis.                   |              | Inparticular,theMed-MoEtakes |             |             |               |           |
|          |        |                               |     |     |     |     | lightweight                      | LLMs         | with                         | smaller     |             | sizes of      | parame-   |
|          |        |                               |     |     |     |     | ters as                          | the base     | model                        | of experts, |             | which         | are first |
|          |        |                               |     |     |     |     | trained                          | with medical |                              | image       | and         | caption       | pairs to  |
|          |        |                               |     |     |     |     | alignvisualandtextualmodalities. |              |                              |             |             | Afterward,the |           |
|          |        |                               |     |     |     |     | model is                         | trained      | with                         | medical     | instruction |               | follow-   |
| Figure1: | Upper: | Thisfigureshowcasesourmodel’s |     |     |     |     |                                  |              |                              |             |             |               |           |
ingdatasetstobetterperformmultimodalmedical
capabilityinaddressingthreeprimarytypesofMedical
|                                           |     |     |     |     |        |     | tasks. Meanwhile,arouteristrainedtoidentifydif- |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | ------ | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- |
| VQAchallengesandimageclassificationtasks. |     |     |     |     | Lower: |     |                                                 |     |     |     |     |     |     |
ferentmedicalimagemodalities,enablingbetterse-
ComparisonbetweenMed-MoEandLLaVA-Med,em-
lectionacrossmultipledomain-specificexpertsdur-
| phasizing | Med-MoE’s | advantages |     | in inference |     | speed, |     |     |     |     |     |     |     |
| --------- | --------- | ---------- | --- | ------------ | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
modelsize,anditssuperiorperformance. ingdecision-making. Inspiredbythewell-known
ResNet(Heetal.,2016)architectureandtheMulti-
|         |          |       |         |     |              |     | Disciplinary | Team | (MDT)   |        | diagnosis | mechanism  |      |
| ------- | -------- | ----- | ------- | --- | ------------ | --- | ------------ | ---- | ------- | ------ | --------- | ---------- | ---- |
| to tune | LLama-7B | model | family, |     | e.g., LLaVA- |     |              |      |         |        |           |            |      |
|         |          |       |         |     |              |     | in clinics,  | we   | propose | to add | an        | additional | meta |
Med. Moreover,themedicaldatadiffersdrastically
expertintheshortcut,asshowninFigure2,which
fromwebcontents,anditsinherentmulti-modality, captures global medical information to assist the
suchasimagingfromCT,MRI,X-rayandpathol- specified expert for better performance. During
ogy,presentsadditionalchallengestodevelopef-
inference,onlythemetaexpertandtheselectedex-
fective yet lightweight medical MLLMs (Acosta pertsareactivated,leadingtoalightweightmodel
etal.,2022;Xuetal.,2024b). Thistaskbecomes withonlyasmallportionofactivatedparameters.
evenmoredifficultwhenconsideringtherequire-
|     |     |     |     |     |     |     | The | Med-MoE | consistently |     | demonstrates |     | sig- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | ------------ | --- | ---- |
mentsforreliabilityandinterpretabilityindecision-
nificantperformanceimprovementsacrossdiverse
making(Salahuddinetal.,2022;Vellido,2020). medical datasets, encompassing both open- and
Recentworkexplorescost-effectivetrainingof close-endMed-VQAandmedicalimageclassifica-
light-weightLLMsbyarchitecturedesign,training
tiontasksinVQA-RAD,SLAKE,PathVQA,Pneu-
| procedure | or hardware |     | optimization |     | etc (Dubiel |     |             |     |                  |     |     |            |     |
| --------- | ----------- | --- | ------------ | --- | ----------- | --- | ----------- | --- | ---------------- | --- | --- | ---------- | --- |
|           |             |     |              |     |             |     | moniaMNIST, |     | and OrganCMNIST. |     |     | Comprehen- |     |
etal.,2024;Zhaoetal.,2024;Huetal.,2024;Zhou siveexperimentsshowthatourMed-MoEs,which
| etal.,2024). | Amongthesetechniques,theMixture- |     |     |     |     |     |                 |     |      |                 |     |       |       |
| ------------ | -------------------------------- | --- | --- | --- | --- | --- | --------------- | --- | ---- | --------------- | --- | ----- | ----- |
|              |                                  |     |     |     |     |     | are constructed |     | with | two small-scale |     | LLMs, | i.e., |
of-Expert(MoE)strategyhasshowngreatpotential
|     |     |     |     |     |     |     | Phi2 (2.7B) | (Abdin |     | et al., | 2024) | and | StableLM |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | --- | ------- | ----- | --- | -------- |
forgeneral-purposetraining (Chenetal.,2022;He (1.7B)(Bellagenteetal.,2024),canattainperfor-
etal.,2021;Jacobsetal.,1991;Eigenetal.,2013), mancesuperiortooronparwiththestate-of-the-art
| e.g., the | Mixtral | family | employs | sparse | MoEs | to  |     |     |     |     |     |     |     |
| --------- | ------- | ------ | ------- | ------ | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
LLaVA-Med(7B)model,withonly2.0-3.6Bacti-
achievecompetingperformancewithLLama-70B
|     |     |     |     |     |     |     | vatedparameters. |     | Extensiveablationsandanalysis |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ----------------------------- | --- | --- | --- | --- |
with only 12.9B active parameters (Jiang et al., demonstrate the efficacy of our Med-MoE in ad-
| 2024); | the MoE-LLaVA |     | propose |     | a MoE-based |     |     |     |     |     |     |     |     |
| ------ | ------------- | --- | ------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
vancingmultimodalmedicaltasksandhighlightits
sparselargeVLMframeworkwithnoveltraining
potentialtoenhanceoutcomesinresource-limited
| strategies(Linetal.,2024). |     |     | Bycombiningmultiple |     |     |     | healthcaresettings. |     |     |     |     |     |     |
| -------------------------- | --- | --- | ------------------- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- |
small-scalesub-modules,i.e.,experts,andactivat-
| ing only | the top-k | relevant | experts |     | for each | task, |           |     |     |     |     |     |     |
| -------- | --------- | -------- | ------- | --- | -------- | ----- | --------- | --- | --- | --- | --- | --- | --- |
|          |           |          |         |     |          |       | 2 Methods |     |     |     |     |     |     |
theMoEmodelcanachievegoodperformancewith
muchlesscomputingcost. Despitethesesuccesses The training of Med-MoE includes three phases,
ingeneraldomains,currentMoEsoftenoverlook asillustratedinFigure2. First,weperformmulti-
thespecializationandsynergyofexpertsrequired modalmedicalalignmenttohelptheLLMtocom-
inmedicalcontexts,andtheirapplicabilityinthe prehendmedicalimagesbyleveragingthevision

Captions
|     |     | C   | Word      | Ttext |     |     |     |     |     |     |     |            |     |
| --- | --- | --- | --------- | ----- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- |
|     |     |     | Embedding |       |     |     |     |     |     |     |     | Activated  |     |
Forward
|     |     |     |         |           | Self- |            |     |     |            |     | Language  |          |                  |
| --- | --- | --- | ------- | --------- | ----- | ---------- | --- | --- | ---------- | --- | --------- | -------- | ---------------- |
|     |     |     |         | Attention |       | Add & Norm |     | FFN | Add & Norm |     | Response  |          |                  |
|     |     | I   | Vision  |           |       |            |     |     |            |     | LAlign    | A        | c t i va t e d   |
|     |     |     | Linear  |           |       |            |     |     |            |     |           | F        | o r w a r d      |
|     |     |     | Encoder | Ti        |       |            |     |     |            |     |           | (Sample) |                  |
Image-Caption pairs
Copy Weight (a) Phase 1 Mutimodel Medical Alignment Non-trainable
| I n s t r uctions   |     | Cinstr |               |        |       |     |     |     |            |     |                  |     |                    |
| ------------------- | --- | ------ | ------------- | ------ | ----- | --- | --- | --- | ---------- | --- | ---------------- | --- | ------------------ |
|                     |     |        | W o r d       | Tinstr |       |     |     | FFN | Add & Norm |     | L a n g u a g e  |     |                    |
| G P T :             |     |        | Em be d d ing |        |       |     |     |     |            |     | R e s p o n s e  | No  | n -a c ti v a ted  |
| H G u P m T : a n : |     |        |               |        |       |     |     |     |            |     | L                |     | F o r w a r d      |
| H.. u. m a n :      |     |        |               |        | Self- |     |     |     |            |     | I n st r         |     |                    |
Add & Norm
Attention
Copy Weight
|     |     | I   | Vision Linear |     |     |     |     | Router |     |     | Domain  |     |     |
| --- | --- | --- | ------------- | --- | --- | --- | --- | ------ | --- | --- | ------- | --- | --- |
|     |     |     | Encoder       | Ti  |     |     |     |        |     |     | Label   |     |     |
LRouter
| Image-Instructions pairs |     |     |     |     | (b) Phase 2 InstructionTuning and Routing |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Copy Weight
Expert 1
| Is the lung  |     | Cinstr | Word | Tinstr |     |     |     |     |     |            |     |     |              |
| ------------ | --- | ------ | ---- | ------ | --- | --- | --- | --- | --- | ---------- | --- | --- | ------------ |
|              |     |        |      |        |     |     |     |     |     | Exp e rt 2 |     | L a | n g u a g e  |
healthy? Embedding Add & Norm Router Add & Norm R e sp o n s e
|     |     |     |     |     | S e lf - |     |     |     |     | .. . |     |     |     |
| --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | ---- | --- | --- | --- |
L M oE
|     |     |     |                   | At  | te n t ion |      |          |                | Expert M       |     |     |     |     |
| --- | --- | --- | ----------------- | --- | ---------- | ---- | -------- | -------------- | -------------- | --- | --- | --- | --- |
|     |     | I   | V i s io n Linear |     |            |      |          | . . .          |                |     |     |     |     |
|     |     |     | E n c o d er      | Ti  |            | M e  | ta MRICT | . . . Domain M | Domain Experts |     |     |     |     |
|     |     |     |                   |     |            | E xp | e rt     |                |                |     |     |     |     |
Domain Image-Instruction pairs
(c) Phase 3 Domain-Specific MoE Tuning
|     |     |     | Figure2: | TheframeworkofMed-MoEwiththreephases. |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | -------- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
encoder’simagetokens. Next,weconductinstruc- selectioninthenextphase. Theinstructiontokens
tiontuningtoenablethemodeltoexecutevarious T and image tokens T are concatenated into
|     |     |     |     |     |     |     | instr |     |     |     | i   |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
medicaltasksandenhanceitsinstruction-following T andfedintotheLLM.Themodelistrained
comb
ability. Meanwhile,arouteristrainedwithasmall usingadatasetofmedicalqueriesandresponsesto
amount of labeled data to characterize the input generateaccurateresponses,minimizingtheloss:
| modality. | Finally, |     | we perform | domain-specific |     |     |     |     |     |     |     |     |     |
| --------- | -------- | --- | ---------- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MoE tuning by replacing the model’s FFN with N (cid:16) (cid:17)
|     |     |     |     |     |     |     |     |         | (cid:88) |      | [ P + i]          | [ :i     | − 1]    |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ---- | ----------------- | -------- | ------- |
|     |     |     |     |     |     |     |     | L Instr | =−       | logp | T r e s p |T comb | ,T r e s | p . (2) |
sparselyactivatedexperts,whereameta-expertis
i=1
alwaysactivatedtocaptureglobalinformation.
Wealsotrainaroutertopredicttheinputmodal-
| 2.1 Phase1: |     | MultimodalMedicalAlignment |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ityusingasmallsubsetofdatawiththeloss:
| In this | phase, | we train | only | the MLP | following |     |     |     |     |     |     |     |     |
| ------- | ------ | -------- | ---- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
M
| thevisionencodertoachievemodalityalignment. |     |     |     |     |     |     |     |     |           | (cid:88) |             |      |     |
| ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | -------- | ----------- | ---- | --- |
|                                             |     |     |     |     |     |     |     | L   | Router =− |          | y logp(y |T | ),   | (3) |
| Wealignvisualandtextualmodalitiesbycurating |     |     |     |     |     |     |     |     |           |          | i i         | comb |     |
i=1
| a dataset | of medical |     | images     | I paired | with      | corre- |                                        |                                     |     |     |     |     |        |
| --------- | ---------- | --- | ---------- | -------- | --------- | ------ | -------------------------------------- | ----------------------------------- | --- | --- | --- | --- | ------ |
|           |            |     |            |          |           |        | wherey                                 | isthetruelabeloftheinputimagemodal- |     |     |     |     |        |
| sponding  | captions   | C.  | The images |          | I are fed | into   |                                        | i                                   |     |     |     |     |        |
|           |            |     |            |          |           |        | ity,i.e.,CT,MRI,PathologyandX-ray,etc. |                                     |     |     |     |     | Inthis |
| a vision  | encoder    | E v | to produce | image    | tokens    | T i    |                                        |                                     |     |     |     |     |        |
(T = E (I )), and the captions C are tokenized phase,thevisionencoderisfrozen,whileallother
| i   | v i |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
componentsaretrained.
| intotexttokensT |     |        | (T =       | Tokenizer(C | )). | The  |     |         |                          |     |     |     |     |
| --------------- | --- | ------ | ---------- | ----------- | --- | ---- | --- | ------- | ------------------------ | --- | --- | --- | --- |
|                 |     | text   | text       |             | i   |      |     |         |                          |     |     |     |     |
| concatenated    |     | tokens | T comb are | fed into    | the | LLM, |     |         |                          |     |     |     |     |
|                 |     |        |            |             |     |      | 2.3 | Phase3: | Domain-SpecificMoETuning |     |     |     |     |
whichistrainedtogeneratethecontinuationoftext
tokens,minimizingtheself-supervisedloss: Finally, we replace the LLM’s FFN with
|     |     |     |          |     |          |     | MoE(mixture-of-experts)architecture. |     |     |     |     | Therouter, |     |
| --- | --- | --- | -------- | --- | -------- | --- | ------------------------------------ | --- | --- | --- | --- | ---------- | --- |
|     |     | N   | (cid:16) |     | (cid:17) |     |                                      |     |     |     |     |            |     |
(cid:88) T[P+i] ,T[:i−1] trainedinphase2,assignsinputstospecificexperts,
| L Align | =−  | logp | text | |T comb | text | . (1) |     |     |     |     |     |     |     |
| ------- | --- | ---- | ---- | ------- | ---- | ----- | --- | --- | --- | --- | --- | --- | --- |
whileameta-expertisalwaysactivatedtocapture
i=1
|             |     |                             |     |     |     |     | global                                   | information. |     | The | MoE layer’s | output | is a |
| ----------- | --- | --------------------------- | --- | --- | --- | --- | ---------------------------------------- | ------------ | --- | --- | ----------- | ------ | ---- |
| 2.2 Phase2: |     | InstructionTuningandRouting |     |     |     |     |                                          |              |     |     |             |        |      |
|             |     |                             |     |     |     |     | weightedcombinationoftheexperts’outputs. |              |     |     |             |        | The  |
Thisphaseaimstoenhancethemodel’sabilityto domain-specific experts and the meta-expert are
followcomplexmedicalinstructionstoperformvar- initialized with the FFN weights from the model
iousmultimodaltasksandtrainarouterforexpert trained in phase 2. In this phase, the router from

phase 2 is used and frozen, so only the domain- Baselines: Wecompareourmethodwithadiverse
specificexpertsandthemeta-expertaretrained. setofbaselines: (1)CLIP-basedmethods,suchas
BiomedCLIPandCLIP-ViT(Zhangetal.,2023b;
K
O MoE = (cid:88) G i E i +Emeta, (4) Eslami et al., 2023), which are state-of-the-art in
i=1 this category but are limited by their reliance on
where G is the gating function provided by the candidatewordsforansweringquestionsinopen
i
router, E are the domain-specific experts, and settings; (2) OFA (One for All)-based models,
i
E isthemeta-expert. Thetraininglossis: liketherecentBiomedGPT(Zhangetal.,2023a),
meta
whichleveragegenerativemultimodalpretraining
L MoE =− (cid:88) N logp (cid:16) T r [ e P s + p i] |O MoE ,T r [ e :i s − p 1] (cid:17) . (5) andhaveshownpromisingperformanceinthemed-
icalfield,buttheirlackofmulti-turndialoguecapa-
i=1
bility,duetonotbeingLLM-based,restrictstheir
Intheend,themodelisfine-tunedforspecificmed-
usageinclinicalpractice;(3)MLLM-basedmod-
icaldomains,leveragingexpertknowledgetopro-
els,includingMed-Flamingoandthestate-of-the-
videhighlyaccurateandrelevantresponsesacross
art LLaVA-Med, which, despite their impressive
theopen-andclose-endandclassificationtasks.
VQAperformance,havelargeparametersizes(7B
3 Experiment and above) that hinder their applicability in real-
worldclinicalsettings. Inclassificationtasks,we
3.1 ExperimentSettings
compare with ViT-based methods and the latest
Dataset: We utilize well-organized datasets pro- Med-Mamba (Yue and Li, 2024). Notably, our
videdbyLLaVA-Med(Lietal.,2024a)foralign- Med-MoE,anMLLM-basedmethod,offersmulti-
mentandinstructiontuninginphase1&2,seede- turn dialogue capabilities for open VQA settings
tailsinSupplementaryFigure12. InMoE-tuning whicharenotpresentintraditionalmethods,while
phase, we employ VQA-RAD (Lau et al., 2018), exhibitingeffectivetraining/inferenceandcompet-
SLAKE (Liu et al., 2021), PathVQA (He et al., ingperformance.
2020)withopen-andclose-endQApairsforMed-
3.2 MainResults
VQAtuningandevaluation. Forclassificationtask,
weusethePneumoniaMNISTandOrganCMNIST Zero-shotPerformanceonMed-VQAtasks: Our
from (Yangetal.,2023). Detailedinformationand modelsexhibitnotableimprovementsinzero-shot
examples of Med-MoE’s responses are shown in performance across various medical VQA tasks.
Supplementary. TheMed-MoE(Phi2)modelboostsscoresbyap-
Evaluation Metrics: We employ the accuracy proximately 1.4% in VQA-RAD Open, 2.6% in
for closed-set questions and recall for open-set VQA-RAD Closed, 5.3% in SLAKE Open, and
questions,beingconsistentwithexistingworklike 9.4%inSLAKEClosedcomparedtoLLaVA-Med
LLaVA-Medforafaircomparison. InTable2,we (LLama7B). The Med-MoE (StableLM) variant
alsoevaluatetheexactmatchandBLEUscoresfor achievesaround6.8%higherinVQA-RADClosed,
comprehensiveevaluation. 5.0% in SLAKE Closed, and 9.3% in PathVQA
Experiment Setup: We select two small LLMs, Closed,demonstratingrobustperformance. These
i.e.,StableLM(1.7B)andPhi2(2.7B),asthebase resultshighlightthesuperioreffectivenessofMed-
model,seeFigure13. Thesizeofactivatedparam- MoEmodelsinzero-shotsettings.
etersinresultingMed-MoEsare2.0B(Med-MoE ComparisonwithSOTAMethodsonMed-VQA:
StableLM) and 3.6B (Med-MoE Phi2), with ad- Overall,Med-MoEcanachievesuperiororcompet-
ditionalparametersfromdomain-specificexperts ingperformancewiththebest-performingLLaVA-
andmeta-experts. Toensureafaircomparisonwith Med(7B)withonly2.0Bor3.6Bactivatedparam-
LLaVA-Med,wealsotrainaLLaVA-Medmodel eters. In particular, Med-MoE (Phi2) surpasses
using the Phi2 (2.7B) backbone. This allows us the best LLaVA-Med variants in SLAKE Open
tocomparetheperformanceunderthesameLLM (85.06), SLAKE Closed (85.58), and PathVQA
backbone. Wealsoinvestigatetheversatilityofour Closed(91.98),andshowscompetingperformance
method by combining it with other cost-efficient on the rest tasks. Med-MoE (StableLM) also ex-
approaches, such as LoRA-based methods. Our hibits better performance than the LLaVA-Med
experimental hyperparameters are shown in Sup- (Phi-2.7B)inmostscenarios,withonly2.0Bacti-
plementaryTable11. vated parameters. Its performance is also on par

|        |     |     |     |     |     |     | VQA-RAD |        |      | SLAKE  | PathVQA |        | Act. |
| ------ | --- | --- | --- | --- | --- | --- | ------- | ------ | ---- | ------ | ------- | ------ | ---- |
| Method |     |     |     |     |     |     | Open    | Closed | Open | Closed | Open    | Closed |      |
Representative&SoTAmethodswithnumbersreportedintheliterature(Non-MLLMBasedMethods)
| VLEncoder–Decoder(Bazietal.,2023) |     |     |     |     |     |     | -   | 82.47 | -   | -   | -     | 85.61 | -   |
| --------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ----- | ----- | --- |
| Q2ATransformer(Liuetal.,2023)     |     |     |     |     |     |     | -   | 81.20 | -   | -   | 54.85 | 88.85 | -   |
PrefixT.MedicalLM(vanSonsbeeketal.,2023) - - - 82.01 - 87.00 -
| PubMedCLIP(Eslamietal.,2023) |     |     |     |     |     |     | -   | 80.00 | -   | 82.50 | -   | -     | -   |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | ----- | --- | ----- | --- |
| BiomedCLIP(Zhangetal.,2023b) |     |     |     |     |     |     | -   | 79.80 | -   | 89.70 | -   | -     | -   |
| M2I2(Lietal.,2022)           |     |     |     |     |     |     | -   | 83.50 | -   | 91.10 | -   | 88.00 | -   |
BiomedGPT-S(Zhangetal.,2023a) 13.40 57.80 66.50 73.30 10.70 84.20 -
BiomedGPT-M(Zhangetal.,2023a) 53.60 65.07 78.30 86.80 12.5 85.70 -
| CLIP-ViTw/GPT2-XL |     |     |     |     |     |     | -   | -   | 84.30 | 82.10 | 40.0 | 87.00 | -   |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ----- | ---- | ----- | --- |
Supervisedfinetuningresults(MLLMBasedMethods)
| LLaVA               |     |     |     |     |     |     | 50.00 | 65.07 | 78.18 | 63.22 | 7.74  | 63.20 | 7B   |
| ------------------- | --- | --- | --- | --- | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | ---- |
| LLaVA-Med(LLama7B)  |     |     |     |     |     |     | 61.52 | 84.19 | 83.08 | 85.34 | 37.95 | 91.21 | 7B   |
| LLaVA-Med(Vicuna7B) |     |     |     |     |     |     | 64.39 | 81.98 | 84.71 | 83.17 | 38.87 | 91.65 | 7B   |
| LLaVA-Med(Phi2.7B)  |     |     |     |     |     |     | 54.83 | 81.35 | 81.29 | 83.29 | 31.73 | 90.17 | 2.7B |
| Med-MoE(Phi2)       |     |     |     |     |     |     | 58.55 | 82.72 | 85.06 | 85.58 | 34.74 | 91.98 | 3.6B |
| Med-MoE(StableLM)   |     |     |     |     |     |     | 50.08 | 80.07 | 83.16 | 83.41 | 33.79 | 91.30 | 2.0B |
Zero-shotresults
| LLaVA-Med(LLama7B) |     |     |     |     |     |     | 36.23 | 60.16 | 41.72 | 47.60 | 10.86 | 59.75 | -   |
| ------------------ | --- | --- | --- | --- | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- |
| Med-MoE(Phi2)      |     |     |     |     |     |     | 36.73 | 61.75 | 43.93 | 56.97 | 6.94  | 66.46 | -   |
| Med-MoE(StableLM)  |     |     |     |     |     |     | 28.02 | 66.91 | 40.63 | 52.64 | 9.40  | 69.09 | -   |
Table1: PerformanceonMed-VQAtasks. Bolddenotesthebestperformance;underlineddenotesthesecond-best.
|        |     |     |     |     |     |     | VQA-RAD |      |     | SLAKE |     | PathVQA |     |
| ------ | --- | --- | --- | --- | --- | --- | ------- | ---- | --- | ----- | --- | ------- | --- |
| Method |     |     |     |     |     | EMS |         | R BS | EMS | R     | BS  | EMS R   | BS  |
LLaVA-Med 7B 58.33 61.52 54.13 82.83 83.08 81.69 37.95 36.86 32.89
Med-Flamingo(Few-Shot)(Mooretal.,2023) 9B 20.00 - - - - - 31.00 - -
| PaLM-E(Tuetal.,2024) |     |     |     |     | 84B |     | -   | - 59.19 | -   | -   | 52.65 | - - | 54.92 |
| -------------------- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | ----- | --- | ----- |
Med-MoE(Phi2) 3.6B 59.69 58.55 52.95 84.46 85.06 83.16 34.37 34.74 32.85
Med-MoE(StableLM) 2.0B 52.53 50.08 45.67 82.44 83.16 81.53 33.60 33.79 32.67
Table2: Detailedcomparisonregardingmoremetrics(SupplementaryA)inOpensettings.
|         |     |     |                |     |             |     |     | outperforming |     | BiomedGPT | and | Med-Mamba. | It  |
| ------- | --- | --- | -------------- | --- | ----------- | --- | --- | ------------- | --- | --------- | --- | ---------- | --- |
| Methods |     |     | PneumoniaMNIST |     | OrganCMNIST |     |     |               |     |           |     |            |     |
Med-Mamba(YueandLi,2024) 91.2 92.4 alsoshowcasesthesecond-bestperformanceonOr-
| AutoKeras(Jinetal.,2019) |     |     |     | 87.8 |     | 87.9 |     |                                      |     |     |     |     |     |
| ------------------------ | --- | --- | --- | ---- | --- | ---- | --- | ------------------------------------ | --- | --- | --- | --- | --- |
|                          |     |     |     |      |     |      |     | ganCMNIST,closelyfollowingMed-Mamba. |     |     |     |     | The |
| BiomedGPT                |     |     |     | 90.8 |     | 88.9 |     |                                      |     |     |     |     |     |
Med-MoE(StableLM) 89.3 88.6 performance of Med-MoE (StableLM) is a little
Med-MoE(Phi2) 91.4 89.9 bitworse. Overall,theclassificationperformance
|         |                                        |     |     |     |     |     |     | ofMed-MoEisquitepromising, |     |     |     | whileitsperfor- |     |
| ------- | -------------------------------------- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | --------------- | --- |
| Table3: | Imageclassificationaccuracycomparison. |     |     |     |     |     |     |                            |     |     |     |                 |     |
mancemightbeboostedifmorerelevantdatarather
thanimage-captionpairscouldbeusedformodel
alignmentandtuningintheinitialphases.
| with LLaVA-Med                      |           |     | (7B) in       | many scenarios, |            | with  |     |                       |     |     |     |     |     |
| ----------------------------------- | --------- | --- | ------------- | --------------- | ---------- | ----- | --- | --------------------- | --- | --- | --- | --- | --- |
| evenbetterperformanceinSLAKEClosed. |           |     |               |                 |            | These |     |                       |     |     |     |     |     |
|                                     |           |     |               |                 |            |       |     | 4 AblationandAnalysis |     |     |     |     |     |
| results                             | highlight | the | effectiveness |                 | and strong | po-   |     |                       |     |     |     |     |     |
tentialofMed-MoEtoestablishnewbenchmarks
| acrossvariousdatasetsandtasks. |            |          |       |                 |       |       |     | Method          |     |     | SFT   | MoE-Tuning   |     |
| ------------------------------ | ---------- | -------- | ----- | --------------- | ----- | ----- | --- | --------------- | --- | --- | ----- | ------------ | --- |
|                                |            |          |       |                 |       |       |     | VQA-RAD(Open)   |     |     | 54.83 | 58.55(+3.72) |     |
| Results                        | on Medical |          | Image | Classification: |       |       | In  |                 |     |     |       | 82.72(+1.37) |     |
|                                |            |          |       |                 |       |       |     | VQA-RAD(Closed) |     |     | 81.35 |              |     |
| contrast                       | to most    | existing |       | LLM-based       | work, | e.g., |     |                 |     |     |       |              |     |
|                                |            |          |       |                 |       |       |     | SLAKE(Open)     |     |     | 81.29 | 85.06(+3.77) |     |
LLaVA-Med, which only evaluates the perfor- SLAKE(Closed) 83.29 85.58(+2.29)
|        |                |     |       |                   |          |      |     | PathVQA(Open)   |     |     | 31.73 | 34.74(+3.01) |     |
| ------ | -------------- | --- | ----- | ----------------- | -------- | ---- | --- | --------------- | --- | --- | ----- | ------------ | --- |
| mance  | on Med-VQA,    |     | we    | further           | evaluate | Med- |     |                 |     |     |       |              |     |
|        |                |     |       |                   |          |      |     | PathVQA(Closed) |     |     | 90.17 | 91.98(+1.81) |     |
| MoE on | classification |     | tasks | for comprehensive |          |      |     |                 |     |     |       |              |     |
analysis. As shown in Table 3, Med-MoE (Phi2) Table4: ComparisonofSFTandMoETuning.
| achieves | 91.4% | accuracy |     | on PneumoniaMNIST, |     |     |     |     |     |     |     |     |     |
| -------- | ----- | -------- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Method NoMetaExpert WithMetaExpert dation compared to LLaVA-Med. For instance,
VQA-RAD(Open) 54.37 58.55(+4.18) in the SLAKE Closed setting, Med-MoE (Phi2)
VQA-RAD(Closed) 81.42 82.72(+1.30) with LoRA exhibits only a 0.49% performance
| SLAKE(Open) |     |     | 81.54 | 85.06(+3.52) |     |               |           |           |         |
| ----------- | --- | --- | ----- | ------------ | --- | ------------- | --------- | --------- | ------- |
|             |     |     |       |              |     | drop, whereas | LLaVA-Med | with LoRA | experi- |
SLAKE(Closed) 82.45 85.58(+3.13) ences1.97%degradation. Furthermore,weobserve
PathVQA(Open) 32.12 34.74(+2.62) that LoRA reduces GPU memory usage during
| PathVQA(Closed) |     |     | 90.19 | 91.98(+1.79) |     |               |             |          |             |
| --------------- | --- | --- | ----- | ------------ | --- | ------------- | ----------- | -------- | ----------- |
|                 |     |     |       |              |     | training, and | our Med-MoE | requires | fewer acti- |
Table5: Ablationonthemetaexpert. vatedparametersduringinference. Consequently,
theintegrationofLoRAwithMed-MoEachieves
lightweightlearningintermsofbothtrainingand
| Method        |     | LearnedRouter |       | Router(Ours) |     |            |                                  |     |     |
| ------------- | --- | ------------- | ----- | ------------ | --- | ---------- | -------------------------------- | --- | --- |
|               |     |               |       |              |     | inference. | ThesefindingsindicatethatMed-MoE |     |     |
| VQA-RAD(Open) |     |               | 56.33 | 58.55(+2.22) |     |            |                                  |     |     |
presentsanappealingpracticalchoiceformedical
| VQA-RAD(Closed) |     |     | 82.19 | 82.72(+0.53) |     |     |     |     |     |
| --------------- | --- | --- | ----- | ------------ | --- | --- | --- | --- | --- |
tasks,deliveringpromisingperformanceatsignifi-
| SLAKE(Open) |     |     | 82.75 | 85.06(+2.31) |     |     |     |     |     |
| ----------- | --- | --- | ----- | ------------ | --- | --- | --- | --- | --- |
cantlylowercomputationalcosts.
| SLAKE(Closed)   |     |     | 84.59 | 85.58(+0.99) |     |                                               |               |              |          |
| --------------- | --- | --- | ----- | ------------ | --- | --------------------------------------------- | ------------- | ------------ | -------- |
|                 |     |     |       |              |     | Effect of                                     | Architectures | and Training | Data of  |
| PathVQA(Open)   |     |     | 33.40 | 34.74(+1.34) |     |                                               |               |              |          |
|                 |     |     |       |              |     | Router: Figure3investigatestheeffectivenessof |               |              |          |
| PathVQA(Closed) |     |     | 91.19 | 91.98(+0.79) |     |                                               |               |              |          |
|                 |     |     |       |              |     | differentMLPstructuresintherouter.            |               |              | Wecanno- |
Table6: Ablationontheroutingmechanism. ticethatcomplicatedMLPs,e.g.,using3MLPlay-
ers,mightnotgiverisetoconsistentimprovements
|          |            |     |             |                |     | andmayevenleadtooverfitting. |     | AsshowninFig-      |           |
| -------- | ---------- | --- | ----------- | -------------- | --- | ---------------------------- | --- | ------------------ | --------- |
| Ablation | of Router: |     | We evaluate | the effective- |     |                              |     |                    |           |
|          |            |     |             |                |     | ure 3, a simple              | MLP | with 1 or 2 layers | can learn |
nessofourroutingmechanismtothegeneralMoE
goodembeddingsoftheinputmodality,resulting
routingmechanismacrossMed-VQAdatasets. Re- in clusters with clear boundaries, as well as high
sultsinTable6withthePhi2.7Bmodelshowthat
|                                          |     |     |     |     |     | accuracyandSilhouettescore. |     | Figure4confirmsef- |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | --------------------------- | --- | ------------------ | --- |
| ourrouterachievesconsistentimprovements. |     |     |     |     | The |                             |     |                    |     |
fectivemodalitydifferentiationwithwell-separated
improvementsonOpensettingsareevenmoreev- embeddingsofimage-textpairspostrouterprocess-
ident than in Closed settings, demonstrating the ing. Moreover,wealsoinvestigatetheeffectiveness
| effectiveness | of  | our routing | mechanism | on  | more |     |     |     |     |
| ------------- | --- | ----------- | --------- | --- | ---- | --- | --- | --- | --- |
ofourrouterwhentrainedwithdifferentnumbers
challengingOpenscenarios.
|     |     |     |     |     |     | ofmodalitylabels. | ResultsinFigure6demonstrate |     |     |
| --- | --- | --- | --- | --- | --- | ----------------- | --------------------------- | --- | --- |
AblationofMetaExpert: Weevaluatetheimpact thatthetrainingofourrouteronlyrequireasmall
of the meta expert with ablation results with the setofmodalitylabelswithoutincurringmuchcom-
| Phi2.7BmodelinTable5. |     |     | Wecannoticethatthe |     |     |     |     |     |     |
| --------------------- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- |
putationalcost.
meta expert can bring consistent and significant Image and Text Specialization in Experts: As
improvements over all Med-VQA settings, with showninFigure4,wevisualizethedomainspeci-
| improvements | of  | 1.30-4.18%. | These | consistent |     |                                         |     |     |      |
| ------------ | --- | ----------- | ----- | ---------- | --- | --------------------------------------- | --- | --- | ---- |
|              |     |             |       |            |     | ficityofexpertswhenprocessingMRIinputs. |     |     | Each |
improvements underscore the critical role of the expertshowsdistinctpreferencesforhandlingtext
meta expert in enhancing the model’s ability to or image information. For example, Expert 1
copewithvariousmultimodalmedicaltasks. mainly handles text data, while Expert 2 has no
| Ablation | of Domain-Specific |     | MoE-Tuning: |     | To  |                               |     |                |     |
| -------- | ------------------ | --- | ----------- | --- | --- | ----------------------------- | --- | -------------- | --- |
|          |                    |     |             |     |     | preferencefortextorimagedata. |     | Expert3focuses |     |
assessthebenefitsoftheMoE-Tuningovertradi- onimagedata,whereasExpert4specializesintext
tionalSupervisedFine-Tuning(SFT),weconduct data. This differentiation highlights the Router’s
anablationstudywiththePhi2.7Bmodel. Results abilitytoenhanceMoEmodelefficiencyandper-
acrossthreeMed-VQAdatasets(Table4)demon- formancebyassigningtaskstosuitableexperts.
stratethattheMoE-Tuningcanleadtobetterperfor- Domain Specialization of Images in Experts:
mance. Theseresultsdemonstratetheeffectiveness Figure5visualizestheactivationstatesofMoEex-
ofourMoEarchitecture,givingrisetobetterper-
|     |     |     |     |     |     | pertsduringinference. |     | Wesample200datapoints |     |
| --- | --- | --- | --- | --- | --- | --------------------- | --- | --------------------- | --- |
formancethantuningadenseFFN. across different modalities and datasets, identify-
Comparison with LoRA-based Methods: We ing the top-1 expert with the most activations in
further investigate the compatibility of our meth- eachMoElayer. Thevisualizationshowsdomain
odswithotherlightweighttechniques,i.e.,LoRA. specializationfordifferentinputimagemodalities:
As shown in Table 7, by integrating with LoRA, Expert1andExpert2forCT,Expert2andExpert
Med-MoEexhibitsmuchlessperformancedegra- 3 for MRI, Expert 4 for Pathology, and Expert 1

|        |     |     | VQA-RAD |        | SLAKE |        |     | PathVQA |        | Act. | Rank |
| ------ | --- | --- | ------- | ------ | ----- | ------ | --- | ------- | ------ | ---- | ---- |
| Method |     |     | Open    | Closed | Open  | Closed |     | Open    | Closed |      |      |
LLaVA-Med(LLama7B)withLoRA 58.22(-3.30) 82.13(-2.06) 81.29(-1.79) 83.37(-1.97) 34.33(-3.62) 90.12(-1.09) 7B 128
LLaVA-Med(Vicuna7B)withLoRA 61.37(-3.02) 80.03(-1.95) 82.02(-2.69) 81.74(-1.43) 36.78(-2.09) 90.67(-0.98) 7B 128
Med-LoRAMoE(Phi2) 58.12(-0.43) 82.35(-0.37) 83.58(-0.12) 84.85(-0.49) 32.62(-0.63) 91.18(-0.80) 3.6B 256
Med-LoRAMoE(Phi2) 57.20(-1.35) 81.75(-0.97) 83.95(-0.35) 84.37(-1.21) 33.03(-0.98) 90.83(-1.15) 3.6B 128
Med-LoRAMoE(StableLM) 47.83(-2.25) 79.04(-1.03) 82.12(-1.04) 82.45(-0.96) 33.28(-1.46) 90.80(-0.89) 2.0B 256
Med-LoRAMoE(StableLM) 45.74(-2.65) 78.31(-1.76) 82.27(-0.89) 83.17(-0.24) 32.20(-2.53) 90.62(-1.08) 2.0B 128
Table7: ComparisonofmodelswithLoRAacrossVQAinopenandclosedsettings. Deltasindicateperformance
changes compared to models without LoRA. The smallest changes are in bold while the second smallest are
underlined.
clinicalsettings.
|     |     |     |     |     | Model                 |     | ModelSizeTrainingGPUInferenceTimeLoadMemory |        |     |     |        |
| --- | --- | --- | --- | --- | --------------------- | --- | ------------------------------------------- | ------ | --- | --- | ------ |
|     |     |     |     |     | LLaVA-Med             |     | 7B                                          | >24GB  |     | 5s  | 20GB   |
|     |     |     |     |     | Med-MoE(Phi2)         |     | 3.6B                                        | 23GB   |     | 3s  | 13.4GB |
|     |     |     |     |     | Med-MoE(StableLM)     |     | 2.0B                                        | 12.5GB |     | 3s  | 10.5GB |
|     |     |     |     |     | Med-LoRAMoE(Phi2)     |     | 3.6B                                        | 13.5GB |     | 3s  | 13.7GB |
|     |     |     |     |     | Med-LoRAMoE(StableLM) |     | 2.0B                                        | 8GB    |     | 3s  | 10.8GB |
Table8:Costefficiencycomparisonofdifferentmodels.
|                      |     |             |                       |             | Top-k(Experts=4) |                                        |     | 1Expert |     | 2Experts(Ours) |     |
| -------------------- | --- | ----------- | --------------------- | ----------- | ---------------- | -------------------------------------- | --- | ------- | --- | -------------- | --- |
|                      |     |             |                       |             | VQA-RAD(Open)    |                                        |     | 46.7    |     | 47.2(+0.5)     |     |
|                      |     |             |                       |             | VQA-RAD(Closed)  |                                        |     | 83.4    |     | 83.8(+0.4)     |     |
|                      |     |             |                       |             | SLAKE(Open)      |                                        |     | 82.1    |     | 82.3(+0.2)     |     |
| Method MLPParameters |     | VQA-RAD     | SLAKE                 | PathVQA     |                  |                                        |     |         |     |                |     |
|                      |     |             |                       |             | SLAKE(Closed)    |                                        |     | 83.2    |     | 84.9(+1.7)     |     |
|                      |     | OpenClosed  | Open ClosedOpenClosed |             |                  |                                        |     |         |     |                |     |
|                      |     |             |                       |             | PathVQA(Open)    |                                        |     | 33.9    |     | 34.1(+0.2)     |     |
| a 0.02MB(MLPx1)      |     | 44.53 76.48 | 81.85 81.78           | 31.05 90.42 |                  |                                        |     |         |     |                |     |
|                      |     |             |                       |             | PathVQA(Closed)  |                                        |     | 90.9    |     | 91.8(+0.9)     |     |
| b 0.02MB(MLPx1)      |     | 45.74 78.31 | 82.27 83.17           | 32.20 90.62 |                  |                                        |     |         |     |                |     |
|                      |     |             |                       |             | Time             |                                        |     | 7h      |     |                | 8h  |
| c 1.00MB(MLPx2)      |     | 45.03 77.64 | 82.36 82.98           | 32.13 90.67 |                  |                                        |     |         |     |                |     |
| d 1.13MB(MLPx3)      |     | 44.98 77.85 | 81.77 82.09           | 31.87 90.17 |                  |                                        |     |         |     |                |     |
|                      |     |             |                       |             | Table9:          | Performancewithvaryingactivatedexperts |     |         |     |                |     |
Figure3: Visualizationoftaskembeddingsandperfor-
manceusingroutersundervariedsettings. Silhouette EffectoftheNumberofActivatedExperts: The
score(sil. score)denotessuperiortaskdifferentiation. router’stop-k selectionistypically1or2inmany
SupplementaryFigure10illustratesPhi2’sembeddings.
|     |     |     |     |     | existing      | MoE works, |          | as more | would | negate    | spar- |
| --- | --- | --- | --- | --- | ------------- | ---------- | -------- | ------- | ----- | --------- | ----- |
|     |     |     |     |     | sity benefits | and        | increase | memory  |       | overhead. | We    |
evaluateperformancewithdifferentnumbersofac-
for X-Ray. This specialization, due to our router tivatedexperts,asshowninTable9. Ultimately,we
andmetaexperts,enhancesMoEperformanceby
chose4expertswithtop-2activationsforbalanced
encouragingeachexperttofocusonspecificmodal-
performanceandoverhead.
itiesandcollaboratewiththemetaexpertforglobal
information. However, visualizing expert activa- Experts(Top-k=2) 2Experts 4Experts(Ours) 6Experts
tions for four modalities handled by traditional VQA-RAD(Open) 47.03(-0.8) 47.83 48.03(+0.2)
|     |     |     |     |     | VQA-RAD(Closed) |     | 77.54(-1.5) |     | 79.04 |     | 78.84(-0.2) |
| --- | --- | --- | --- | --- | --------------- | --- | ----------- | --- | ----- | --- | ----------- |
routersineachMoElayerrevealsfusedpatterns,re-
|     |     |     |     |     | SLAKE(Open) |     | 81.52(-0.6) |     | 82.12 |     | 82.42(+0.3) |
| --- | --- | --- | --- | --- | ----------- | --- | ----------- | --- | ----- | --- | ----------- |
sultinginweakerinterpretabilityandperformance.
|     |     |     |     |     | SLAKE(Closed) |     | 81.45(-1.0) |     | 82.45 |     | 82.65(+0.2) |
| --- | --- | --- | --- | --- | ------------- | --- | ----------- | --- | ----- | --- | ----------- |
Cost Analysis: Table 8 illustrates the cost effi- PathVQA(Open) 32.78(-0.5) 33.28 33.58(+0.3)
|           |            |          |               |     | PathVQA(Closed) |     | 90.60(-0.2) |     | 90.80 |     | 91.10(+0.3) |
| --------- | ---------- | -------- | ------------- | --- | --------------- | --- | ----------- | --- | ----- | --- | ----------- |
| ciency of | our models | compared | to LLaVA-Med. |     |                 |     |             |     |       |     |             |
|           |            |          |               |     | Time            |     | 6h          |     | 8h    |     | 11h         |
Particularly,Med-LoRAMoEmodelsshowsignif-
icant reductions in training GPU memory usage Table10: Performancewithvariedexpertnumber
| andinferencetime. |     | Forexample,Med-LoRAMoE |     |     |     |     |     |     |     |     |     |
| ----------------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(StableLM) requires only 8GB of GPU memory Effect of the Number of Experts: In MoE ap-
and3secondsforinference,demonstratinghighef- plications, choosing the right number of experts
ficiencyfordeployment,andmakingourapproach iscrucialforbalancingperformanceandcomputa-
more appealing to practical resource-constrained tionalefficiency. Weevaluateconfigurationswith

Figure4:VisualizationofexpertspecializationinprocessingimageandtexttokensundertheMRImodality. Results
forothermodalitiesareinSupplementaryFigure11.
Figure5: Upper: ExpertactivationsforfourmodalitieshandledbyourrouterofMed-MoEineachMoElayer.
Lower: ExpertactivationsforfourmodalitieshandledbythestandardlearnedrouterineachMoElayer.
2, 4, and 6 experts, each with t h e top-2 activa- etal.,20 2 4 a;Chenetal.,2024;Gouetal.,2023)
tions. ResultsinTable10showthatusing4experts uses Top - 1 activation to assign different tasks to
achievesthebestbalancebetweenperformanceand different experts, avoiding performance degrada-
costefficiency. Whileincreasingto6expertsoffers tionfromtaskdataconflictsbutoverlookingmodal
(a). Expert activations for four modalities handled by the standard learned router in each MoE layer.
slightperformancegains,itsignificantlyincreases biases within the same task type. The second
(b). Expert activations for four modalities handled by our router of Tinymed-MoE in each MoE layer.
computational time and memory usage. On the approach (Lin et al., 2024; Li et al., 2024b; Lee
otherhand,reducingthenumberto2expertsleads etal.,2024;Liuetal.,2024b;Daietal.,2024)re-
todecreasedperformanceacrossalltasks. placesFFNlayersinLLMswithMoEstructuresus-
ingmultipleexpertactivations,achievingimprove-
5 RelatedWork
mentswithminimaladditionalparameters. How-
ever, visualizing the expert activations in routers
Medical MLLMs Advancements in Medical
shows these methods often fail to specialize ex-
MLLMs, such as Med-Flamingo (Moor et al.,
pertseffectively,limitinginterpretabilityandper-
2023), Med-PaLM M (Singhal et al., 2023), and
formancewithdiversedatamodalities(Fanetal.,
LLaVA-Med(Lietal.,2024a),havesignificantly
2024). AspecializedMoEarchitecturetailoredto
impacted medical diagnostics and patient care,
themedicaldomainisneededtoleveragemodality-
buildingongeneralAImodelslikeChatGPT (Ope-
specificinformationandimproveperformancewith
nAI,2022)andGPT-4(OpenAI,2023). Thesemod-
asmallerLLMbackbone.
els enhance few-shot learning, medical question
answering,andconversationalAI,demonstrating
6 Conclusion
thepotentialofspecializedMLLMsinhealthcare.
Biomedicalchatbots like ChatDoctor (Yunxiang We have introduced Med-MoE, a lightweight
etal.,2023)andVisualMed-Alpacahighlightthe frameworkformultimodalmedicaltasks,address-
benefitsofdomain-specificfine-tuning. However, ingbothdiscriminativeandgenerativeneeds. Opti-
theirapplicationinresource-constrainedhospital mizedforresource-constrainedenvironments,Med-
settings remains underexplored, emphasizing the MoE involves aligning medical images with lan-
needforcost-efficientMLLMsinclinicalcontexts. guagemodeltokens,task-specificinstructiontun-
MoEinMLLMsMoEinMLLMsaddressestask ing, and domain-specific expert fine-tuning. Our
conflicts in multi-task learning and offers a cost- approachreducesactivatedparameterswhilemain-
efficient scaling method. The first approach (Xu tainingorsurpassingstate-of-the-artperformance.

OurexperimentsonVQA-RAD,SLAKE,andPath- NguyenBach,AmitBahree,ArashBakhtiari,Harki-
VQA validate Med-MoE’s effectiveness and ef- ratBehl,etal.2024. Phi-3technicalreport: Ahighly
|           |            |     |        |             |     |          | capablelanguagemodellocallyonyourphone. |     |     |     |     |     | arXiv |
| --------- | ---------- | --- | ------ | ----------- | --- | -------- | --------------------------------------- | --- | --- | --- | --- | --- | ----- |
| ficiency. | This model |     | offers | a practical |     | solution |                                         |     |     |     |     |     |       |
preprintarXiv:2404.14219.
fordeployingadvancedmedicalAIindiverseand
resource-limitedclinicalsettings. Julián N Acosta, Guido J Falcone, Pranav Rajpurkar,
|     |     |     |     |     |     |     | andEricJTopol.2022. |     |     | Multimodalbiomedicalai. |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | ----------------------- | --- | --- | --- |
NatureMedicine,28(9):1773–1784.
7 DiscussionandLimitations
StanislawAntol,AishwaryaAgrawal,JiasenLu,Mar-
| Our work | primarily | sought | to  | develop | a   | smaller, |     |     |     |     |     |     |     |
| -------- | --------- | ------ | --- | ------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
garetMitchell,DhruvBatra,CLawrenceZitnick,and
more cost-efficient Multimodal Large Language DeviParikh.2015. Vqa: Visualquestionanswering.
InProceedingsoftheIEEEinternationalconference
| Model (MLLM) |     | for the | medical | field, | diverging |     |     |     |     |     |     |     |     |
| ------------ | --- | ------- | ------- | ------ | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
oncomputervision,pages2425–2433.
| from the | current | trend | focused | on creating |     | larger |     |     |     |     |     |     |     |
| -------- | ------- | ----- | ------- | ----------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
and more robust models. We posit that in practi- Yakoub Bazi, Mohamad Mahmoud Al Rahhal, Laila
|     |     |     |     |     |     |     | Bashmal, | and | Mansour | Zuair. | 2023. | Vision– |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | ------- | ------ | ----- | ------- | --- |
calapplications,especiallyinresource-constrained
|     |     |     |     |     |     |     | language | model | for | visual | question | answering | in  |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | --- | ------ | -------- | --------- | --- |
environmentslikemobiledevices,smallermodels medicalimagery. Bioengineering.
| could be | more advantageous. |     |     | This | approach | not |     |     |     |     |     |     |     |
| -------- | ------------------ | --- | --- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
MarcoBellagente,JonathanTow,DakotaMahan,Duy
onlyaddressesthepracticallimitationsofdeploy-
Phung,MaksymZhuravinskyi,ReshinthAdithyan,
inglarge-scalemodelsinroutineclinicalsettings
|     |     |     |     |     |     |     | James | Baicoianu, | Ben | Brooks, | Nathan | Cooper, |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | ---------- | --- | ------- | ------ | ------- | --- |
butalsoexploresthefeasibilityofusingleanermod- Ashish Datta, et al. 2024. Stable lm 2 1.6 b tech-
elswithoutcompromisingonperformance,foster- nicalreport. arXivpreprintarXiv:2402.17834.
ingbroaderaccessibilityandapplication.
JunChen,DeyaoZhu,XiaoqianShen,XiangLi,Zechun
However,ourapproachfacesseverallimitations. Liu,PengchuanZhang,RaghuramanKrishnamoor-
thi,VikasChandra,YunyangXiong,andMohamed
First,thereisanotablescarcityoftrainingdatain
|     |     |     |     |     |     |     | Elhoseiny.2023. |     | Minigpt-v2: |     | largelanguagemodel |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ----------- | --- | ------------------ | --- | --- |
themedicaldomain,largelyduetothesensitivity
asaunifiedinterfaceforvision-languagemulti-task
andprivacyconcernsassociatedwithmedicaldata.
|            |           |      |         |     |         |      | learning. | arXivpreprintarXiv:2310.09478. |     |     |     |     |     |
| ---------- | --------- | ---- | ------- | --- | ------- | ---- | --------- | ------------------------------ | --- | --- | --- | --- | --- |
| Generating | synthetic | data | through |     | methods | like |           |                                |     |     |     |     |     |
thoseusedforGPT-4Vcanbeproblematicinthis ShaoxiangChen,ZequnJie,andLinMa.2024. Llava-
mole: Sparsemixtureofloraexpertsformitigating
context,andmanydatasetsrequirelabor-intensive
|     |     |     |     |     |     |     | dataconflictsininstructionfinetuningmllms. |     |     |     |     |     | arXiv |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | ----- |
manualannotationsbymedicalprofessionals. This preprintarXiv:2401.16160.
isbothcostlyandlimitsthescalabilityofdatagen-
|         |          |                |     |                  |     |     | Zixiang Chen, | Yihe | Deng, | Yue     | Wu, Quanquan  |     | Gu, |
| ------- | -------- | -------------- | --- | ---------------- | --- | --- | ------------- | ---- | ----- | ------- | ------------- | --- | --- |
| eration | efforts. | As illustrated |     | in Supplementary |     |     |               |      |       |         |               |     |     |
|         |          |                |     |                  |     |     | and Yuanzhi   | Li.  | 2022. | Towards | understanding |     | the |
Figure9,ourmodeloccasionallyfails,particularly mixture-of-expertslayerindeeplearning. Advances
withmorecomplexopen-endedquestionsthatde- inneuralinformationprocessingsystems,35:23049–
| mandprecisemedicalknowledge. |     |     |     |     |     |     | 23062. |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
Furthermore,theinherentrequirementformed-
|     |     |     |     |     |     |     | KateCrawford.2021. |     | TheatlasofAI:Power,politics, |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ---------------------------- | --- | --- | --- | --- |
ical applications to provide trustworthy explana- andtheplanetarycostsofartificialintelligence. Yale
UniversityPress.
| tions and | confidence | scores |     | poses | another | chal- |     |     |     |     |     |     |     |
| --------- | ---------- | ------ | --- | ----- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- |
lenge. Ensuringthatthemodeloutputsarenotonly
DamaiDai,ChengqiDeng,ChenggangZhao,RXXu,
accuratebutalsoaccompaniedbyreliablejustifica- Huazuo Gao, Deli Chen, Jiashi Li, Wangding
|     |     |     |     |     |     |     | Zeng, Xingkai |     | Yu, Y | Wu, et | al. 2024. | Deepseek- |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----- | ------ | --------- | --------- | --- |
tionsiscrucial,especiallyinafieldwheredecisions
|     |     |     |     |     |     |     | moe: Towards |     | ultimate | expert | specialization |     | in  |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | -------- | ------ | -------------- | --- | --- |
havesignificanthealthimplications. Thisnecessity mixture-of-expertslanguagemodels. arXivpreprint
heightenstheimportanceofbuildingatrustworthy
arXiv:2401.06066.
MLLMthatcanarticulateitsreasoningprocesses
AnaCláudiaAkemiMatsukideFaria,FelypedeCas-
clearlyandprovideconfidencelevels,therebyen-
|     |     |     |     |     |     |     | tro Bastos, | José | Victor | Nogueira | Alves | da  | Silva, |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ------ | -------- | ----- | --- | ------ |
hancingthereliabilityandsafetyofAIapplications
|     |     |     |     |     |     |     | Vitor Lopes | Fabris, | Valeska |     | de Sousa | Uchoa, | Dé- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ------- | --- | -------- | ------ | --- |
inhealthcare. cio Gonçalves de Aguiar Neto, and Claudio Filipi
|     |     |     |     |     |     |     | GoncalvesdosSantos.2023. |     |     |     | Visualquestionanswer- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --------------------- | --- | --- |
ing: Asurveyontechniquesandcommontrendsin
Acknowledgments
|     |     |     |     |     |     |     | recentliterature. |     | arXivpreprintarXiv:2305.11033. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------------------------------ | --- | --- | --- | --- |
MateuszDubiel,YasmineBarghouti,KristinaKudryavt-
References
|     |     |     |     |     |     |     | seva,andLuisALeiva.2024. |     |     |     | On-devicequeryintent |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | -------------------- | --- | --- |
MarahAbdin,SamAdeJacobs,AmmarAhmadAwan, prediction with lightweight llms to support ubiqui-
Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, tousconversations. ScientificReports,14(1):12731.

DavidEigen,Marc’AurelioRanzato,andIlyaSutskever. HaifengJin,QingquanSong,andXiaHu.2019. Auto-
2013. Learning factored representations in a deep keras: Anefficientneuralarchitecturesearchsystem.
mixtureofexperts. arXivpreprintarXiv:1312.4314. InProceedingsofthe25thACMSIGKDDinterna-
|     |     |     |     |     |     | tional conference |     | on  | knowledge | discovery |     | & data |
| --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --------- | --------- | --- | ------ |
Sedigheh Eslami, Christoph Meinel, and Gerard mining,pages1946–1956.
| DeMelo.2023. |     | Pubmedclip: | Howmuchdoesclip |     |     |     |     |     |     |     |     |     |
| ------------ | --- | ----------- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
benefitvisualquestionansweringinthemedicaldo- ChristopherJKelly,AlanKarthikesalingam,Mustafa
main? InFindingsoftheAssociationforComputa- Suleyman,GregCorrado,andDominicKing.2019.
|                    |      |                          |     |            |        | Key challenges          |     | for delivering |                     | clinical | impact | with |
| ------------------ | ---- | ------------------------ | --- | ---------- | ------ | ----------------------- | --- | -------------- | ------------------- | -------- | ------ | ---- |
| tionalLinguistics: |      | EACL2023,pages1151–1163. |     |            |        |                         |     |                |                     |          |        |      |
|                    |      |                          |     |            |        | artificialintelligence. |     |                | BMCmedicine,17:1–9. |          |        |      |
| Dongyang           | Fan, | Bettina Messmer,         |     | and Martin | Jaggi. |                         |     |                |                     |          |        |      |
2024. Towardsanempiricalunderstandingofmoe JasonJLau,SoumyaGayen,AsmaBenAbacha,and
arXivpreprintarXiv:2402.13089. Dina Demner-Fushman. 2018. A dataset of clini-
designchoices.
callygeneratedvisualquestionsandanswersabout
|           |        |              |         |               |              | radiologyimages. |      | Scientificdata,5(1):1–10. |     |       |      |          |
| --------- | ------ | ------------ | ------- | ------------- | ------------ | ---------------- | ---- | ------------------------- | --- | ----- | ---- | -------- |
| Peng Gao, | Renrui | Zhang, Chris |         | Liu, Longtian | Qiu,         |                  |      |                           |     |       |      |          |
| Siyuan    | Huang, | Weifeng Lin, | Shitian |               | Zhao, Shijie |                  |      |                           |     |       |      |          |
|           |        |              |         |               |              | Byung-Kwan       | Lee, | Beomchan                  |     | Park, | Chae | Won Kim, |
Geng, Ziyi Lin, Peng Jin, et al. 2024. Sphinx- andYongManRo.2024. Moai: Mixtureofallintel-
| x: Scaling                      | data | and parameters |     | for           | a family of |                                         |     |     |     |     |     |       |
| ------------------------------- | ---- | -------------- | --- | ------------- | ----------- | --------------------------------------- | --- | --- | --- | --- | --- | ----- |
|                                 |      |                |     |               |             | ligenceforlargelanguageandvisionmodels. |     |     |     |     |     | arXiv |
| multi-modallargelanguagemodels. |      |                |     | arXivpreprint |             |                                         |     |     |     |     |     |       |
preprintarXiv:2403.07508.
arXiv:2402.05935.
|     |     |     |     |     |     | Chunyuan | Li, | Cliff Wong, |     | Sheng | Zhang, | Naoto |
| --- | --- | --- | --- | --- | --- | -------- | --- | ----------- | --- | ----- | ------ | ----- |
YunhaoGou,ZhiliLiu,KaiChen,LanqingHong,Hang Usuyama,HaotianLiu,JianweiYang,TristanNau-
Xu,AoxueLi,Dit-YanYeung,JamesTKwok,and mann, Hoifung Poon, and Jianfeng Gao. 2024a.
| YuZhang.2023.                               |     | Mixtureofcluster-conditionallora |     |     |       |            |                 |     |         |                     |          |     |
| ------------------------------------------- | --- | -------------------------------- | --- | --- | ----- | ---------- | --------------- | --- | ------- | ------------------- | -------- | --- |
|                                             |     |                                  |     |     |       | Llava-med: | Training        |     | a large | language-and-vision |          |     |
| expertsforvision-languageinstructiontuning. |     |                                  |     |     | arXiv |            |                 |     |         |                     |          |     |
|                                             |     |                                  |     |     |       | assistant  | for biomedicine |     | in      | one day.            | Advances | in  |
preprintarXiv:2312.12379.
NeuralInformationProcessingSystems,36.
YashGoyal,AkritMohapatra,DeviParikh,andDhruv PengfeiLi,GangLiu,LinTan,JinyingLiao,andShen-
Batra. 2016. Towards transparent ai systems: In- jun Zhong. 2022. Self-supervised vision-language
| terpretingvisualquestionansweringmodels. |     |     |     |     | arXiv |                                |     |         |        |          |     |            |
| ---------------------------------------- | --- | --- | --- | --- | ----- | ------------------------------ | --- | ------- | ------ | -------- | --- | ---------- |
|                                          |     |     |     |     |       | pretraining                    | for | medical | visual | question |     | answering. |
| preprintarXiv:1608.08974.                |     |     |     |     |       | arXivpreprintarXiv:2211.13594. |     |         |        |          |     |            |
JiaaoHe,JiezhongQiu,AohanZeng,ZhilinYang,Ji- Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue
dong Zhai, and Jie Tang. 2021. Fastmoe: A fast Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, and
|                   |     |          |         |       |          | MinZhang.2024b. |     | Uni-moe: |     | Scalingunifiedmulti- |     |     |
| ----------------- | --- | -------- | ------- | ----- | -------- | --------------- | --- | -------- | --- | -------------------- | --- | --- |
| mixture-of-expert |     | training | system. | arXiv | preprint |                 |     |          |     |                      |     |     |
arXiv:2103.13262. modalllmswithmixtureofexperts. arXivpreprint
arXiv:2405.11273.
KaimingHe,XiangyuZhang,ShaoqingRen,andJian
|           |                                    |     |          |            |     | Weibin Liao, | Yinghao |       | Zhu, Xinyuan |         | Wang,       | Cehng-    |
| --------- | ---------------------------------- | --- | -------- | ---------- | --- | ------------ | ------- | ----- | ------------ | ------- | ----------- | --------- |
| Sun.2016. | Deepresiduallearningforimagerecog- |     |          |            |     |              |         |       |              |         |             |           |
|           |                                    |     |          |            |     | wei Pan,     | Yasha   | Wang, | and          | Liantao |             | Ma. 2024. |
| nition.   | In Proceedings                     | of  | the IEEE | conference | on  |              |         |       |              |         |             |           |
|           |                                    |     |          |            |     | Lightm-unet: |         | Mamba | assists      | in      | lightweight | unet      |
computervisionandpatternrecognition,pages770–
|      |     |     |     |     |     |             |       |               |     |     | arXiv | preprint |
| ---- | --- | --- | --- | --- | --- | ----------- | ----- | ------------- | --- | --- | ----- | -------- |
| 778. |     |     |     |     |     | for medical | image | segmentation. |     |     |       |          |
arXiv:2403.05246.
XuehaiHe,YichenZhang,LuntianMou,EricXing,and
|                  |     |          |                    |     |     | Bin Lin, Zhenyu |     | Tang, | Yang Ye, | Jiaxi | Cui, | Bin Zhu, |
| ---------------- | --- | -------- | ------------------ | --- | --- | --------------- | --- | ----- | -------- | ----- | ---- | -------- |
| PengtaoXie.2020. |     | Pathvqa: | 30000+questionsfor |     |     |                 |     |       |          |       |      |          |
PengJin,JunwuZhang,MunanNing,andLiYuan.
| medicalvisualquestionanswering. |     |     |     | arXivpreprint |     |                  |     |                                 |     |     |     |     |
| ------------------------------- | --- | --- | --- | ------------- | --- | ---------------- | --- | ------------------------------- | --- | --- | --- | --- |
|                                 |     |     |     |               |     | 2024. Moe-llava: |     | Mixtureofexpertsforlargevision- |     |     |     |     |
arXiv:2003.10286.
arXivpreprintarXiv:2401.15947.
languagemodels.
WenboHu,YifanXu,YiLi,WeiyueLi,ZeyuanChen,
BoLiu,Li-MingZhan,LiXu,LinMa,YanYang,and
| andZhuowenTu.2024. |     | Bliva: | Asimplemultimodal |     |     |                   |     |     |        |                       |     |     |
| ------------------ | --- | ------ | ----------------- | --- | --- | ----------------- | --- | --- | ------ | --------------------- | --- | --- |
|                    |     |        |                   |     |     | Xiao-MingWu.2021. |     |     | Slake: | Asemantically-labeled |     |     |
llmforbetterhandlingoftext-richvisualquestions.
knowledge-enhanceddatasetformedicalvisualques-
InProceedingsoftheAAAIConferenceonArtificial
|     |     |     |     |     |     | tion answering. |     | In 2021 | IEEE | 18th | International |     |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ------- | ---- | ---- | ------------- | --- |
Intelligence,volume38,pages2256–2264. Symposium on Biomedical Imaging (ISBI), pages
1650–1654.IEEE.
RobertAJacobs,MichaelIJordan,StevenJNowlan,
| andGeoffreyEHinton.1991. |     |     | Adaptivemixturesof |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
HaotianLiu,ChunyuanLi,QingyangWu,andYongJae
localexperts. Neuralcomputation,3(1):79–87. Lee.2024a. Visualinstructiontuning. Advancesin
neuralinformationprocessingsystems,36.
| Albert Q | Jiang, | Alexandre | Sablayrolles, |     | Antoine |     |     |     |     |     |     |     |
| -------- | ------ | --------- | ------------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
Roux,ArthurMensch,BlancheSavary,ChrisBam- YunyiLiu,ZhanyuWang,DongXu,andLupingZhou.
ford,DevendraSinghChaplot,DiegodelasCasas, 2023. Q2atransformer: Improving medical vqa
Emma Bou Hanna, Florian Bressand, et al. 2024. via an answer querying decoder. arXiv preprint
| Mixtralofexperts. |     | arXivpreprintarXiv:2401.04088. |     |     |     | arXiv:2304.01611. |     |     |     |     |     |     |
| ----------------- | --- | ------------------------------ | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |

Zhili Liu, Yunhao Gou, KaiChen, Lanqing Hong, Ji- TaoTu,ShekoofehAzizi,DannyDriess,MikeSchaek-
ahuiGao,FeiMi,YuZhang,ZhenguoLi,XinJiang, ermann,MohamedAmin,Pi-ChuanChang,Andrew
Qun Liu, et al. 2024b. Mixture of insightful ex- Carroll,CharlesLau,RyutaroTanno,IraKtena,etal.
perts (mote): The synergy of thought chains and 2024. Towardsgeneralistbiomedicalai. NEJMAI,
| expert mixtures |     | in self-alignment. |     | arXiv | preprint | 1(3):AIoa2300138. |     |     |     |     |     |
| --------------- | --- | ------------------ | --- | ----- | -------- | ----------------- | --- | --- | --- | --- | --- |
arXiv:2405.00557.
|     |     |     |     |     |     | Tom van | Sonsbeek, | Mohammad |     | Mahdi Derakhshani, |     |
| --- | --- | --- | --- | --- | --- | ------- | --------- | -------- | --- | ------------------ | --- |
YadongLu,ChunyuanLi,HaotianLiu,JianweiYang, Ivona Najdenkoska, Cees GM Snoek, and Marcel
Jianfeng Gao, and Yelong Shen. 2023. An empiri- Worring.2023. Open-endedmedicalvisualquestion
calstudyofscalinginstruct-tunedlargemultimodal answeringthroughprefixtuningoflanguagemodels.
models. arXivpreprintarXiv:2309.09958. arXivpreprintarXiv:2303.05977.
YalinMiao,ShuyunHe,WenFangCheng,GuodongLi,
|     |     |     |     |     |     | AlfredoVellido.2020. |     | Theimportanceofinterpretabil- |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------------- | --- | ----------------------------- | --- | --- | --- |
andMengTong.2022. Researchonvisualquestion ityandvisualizationinmachinelearningforapplica-
answeringbasedondynamicmemorynetworkmodel tionsinmedicineandhealthcare. Neuralcomputing
ofmultipleattentionmechanisms. ScientificReports, andapplications,32(24):18069–18083.
12(1):16758.
|               |      |        |         |     |           | Weihan Wang, |     | Qingsong | Lv, Wenmeng |     | Yu, Wenyi |
| ------------- | ---- | ------ | ------- | --- | --------- | ------------ | --- | -------- | ----------- | --- | --------- |
| Michael Moor, | Qian | Huang, | Shirley | Wu, | Michihiro |              |     |          |             |     |           |
Hong,JiQi,YanWang,JunhuiJi,ZhuoyiYang,Lei
| Yasunaga,                                     | Yash       | Dalmia,  | Jure | Leskovec,  | Cyril Za-  |                                  |     |     |     |               |           |
| --------------------------------------------- | ---------- | -------- | ---- | ---------- | ---------- | -------------------------------- | --- | --- | --- | ------------- | --------- |
|                                               |            |          |      |            |            | Zhao,XixuanSong,etal.2023.       |     |     |     | Cogvlm:       | Visualex- |
| kka, Eduardo                                  | Pontes     | Reis,    | and  | Pranav     | Rajpurkar. |                                  |     |     |     |               |           |
|                                               |            |          |      |            |            | pertforpretrainedlanguagemodels. |     |     |     | arXivpreprint |           |
| 2023. Med-flamingo:amultimodalmedicalfew-shot |            |          |      |            |            | arXiv:2311.03079.                |     |     |     |               |           |
| learner.                                      | In Machine | Learning |      | for Health | (ML4H),    |                                  |     |     |     |               |           |
pages353–367.PMLR. Jingwei Xu, Junyu Lai, and Yunpeng Huang. 2024a.
|               |          |     |                     |     |     | Meteora:     | Multiple-tasksembeddedloraforlargelan- |                                |     |     |     |
| ------------- | -------- | --- | ------------------- | --- | --- | ------------ | -------------------------------------- | ------------------------------ | --- | --- | --- |
| OpenAI. 2022. | ChatGPT. |     | https://openai.com/ |     |     |              |                                        |                                |     |     |     |
|               |          |     |                     |     |     | guagemodels. |                                        | arXivpreprintarXiv:2405.13053. |     |     |     |
blog/chatgpt/.
XiXu,JianqiangLi,ZhichaoZhu,LinnaZhao,Huina
| OpenAI.                           | 2023. |     | GPT-4 | technical | report.   |         |          |         |             |              |       |
| --------------------------------- | ----- | --- | ----- | --------- | --------- | ------- | -------- | ------- | ----------- | ------------ | ----- |
|                                   |       |     |       |           |           | Wang,   | Changwei | Song,   | Yining      | Chen, Qing   | Zhao, |
| https://arxiv.org/abs/2303.08774. |       |     |       |           | Preprint, |         |          |         |             |              |       |
|                                   |       |     |       |           |           | Jijiang | Yang,    | and Yan | Pei. 2024b. | A comprehen- |       |
arXiv:2303.08774.
|                 |        |          |     |        |             | sive review                     | on  | synergy | of multi-modal |                 | data and ai |
| --------------- | ------ | -------- | --- | ------ | ----------- | ------------------------------- | --- | ------- | -------------- | --------------- | ----------- |
|                 |        |          |     |        |             | technologiesinmedicaldiagnosis. |     |         |                | Bioengineering, |             |
| Lena Petersson, | Ingrid | Larsson, |     | Jens M | Nygren, Per |                                 |     |         |                |                 |             |
11(3):219.
Nilsen,MargitNeher,JulieEReed,DanielTyskbo,
| andPetraSvedberg.2022.                 |       |                 | Challengestoimplement- |         |              |             |                                          |                 |       |           |           |
| -------------------------------------- | ----- | --------------- | ---------------------- | ------- | ------------ | ----------- | ---------------------------------------- | --------------- | ----- | --------- | --------- |
|                                        |       |                 |                        |         |              | Fuzhao Xue, | Yao                                      | Fu, Wangchunshu |       | Zhou,     | Zangwei   |
| ingartificialintelligenceinhealthcare: |       |                 |                        |         | aqualitative |             |                                          |                 |       |           |           |
|                                        |       |                 |                        |         |              | Zheng,      | and Yang                                 | You.            | 2024. | To repeat | or not to |
| interview                              | study | with healthcare |                        | leaders | in sweden.   |             |                                          |                 |       |           |           |
|                                        |       |                 |                        |         |              | repeat:     | Insightsfromscalingllmundertoken-crisis. |                 |       |           |           |
BMCHealthServicesResearch,22(1):850.
AdvancesinNeuralInformationProcessingSystems,
36.
ZohaibSalahuddin,HenryCWoodruff,AvishekChat-
| terjee, and | Philippe | Lambin. |     | 2022. | Transparency |     |     |     |     |     |     |
| ----------- | -------- | ------- | --- | ----- | ------------ | --- | --- | --- | --- | --- | --- |
JianchengYang,RuiShi,DonglaiWei,ZequanLiu,Lin
ofdeepneuralnetworksformedicalimageanalysis:
Zhao,BilianKe,HanspeterPfister,andBingbingNi.
| Areviewofinterpretabilitymethods. |     |     |     |     | Computersin |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- |
biologyandmedicine,140:105111. 2023. Medmnistv2-alarge-scalelightweightbench-
markfor2dand3dbiomedicalimageclassification.
Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie ScientificData,10(1):41.
Neiswanger,JoelHestness,NataliaVassilieva,Daria
|                                            |     |     |     |                |       | YubiaoYueandZhenzhangLi.2024.           |     |     |     | Medmamba: | Vi-   |
| ------------------------------------------ | --- | --- | --- | -------------- | ----- | --------------------------------------- | --- | --- | --- | --------- | ----- |
| Soboleva,andEricXing.2023.                 |     |     |     | Slimpajama-dc: | Un-   |                                         |     |     |     |           |       |
|                                            |     |     |     |                |       | sionmambaformedicalimageclassification. |     |     |     |           | arXiv |
| derstandingdatacombinationsforllmtraining. |     |     |     |                | arXiv |                                         |     |     |     |           |       |
preprintarXiv:2403.03849.
preprintarXiv:2309.10818.
BaifengShi,ZiyangWu,MaolinMao,XinWang,and LiYunxiang,LiZihan,ZhangKai,DanRuilong,and
TrevorDarrell.2024. Whendowenotneedlarger ZhangYou.2023. Chatdoctor:Amedicalchatmodel
visionmodels? arXivpreprintarXiv:2403.13043. fine-tuned on llama model using medical domain
|                |     |           |           |     |              | knowledge. | arXivpreprintarXiv:2303.14070. |     |     |     |     |
| -------------- | --- | --------- | --------- | --- | ------------ | ---------- | ------------------------------ | --- | --- | --- | --- |
| Karan Singhal, | Tao | Tu, Juraj | Gottweis, |     | Rory Sayres, |            |                                |     |     |     |     |
KaiZhang,JunYu,ZhilingYan,YixinLiu,EashanAd-
| Ellery Wulczyn, |     | Le Hou, | Kevin | Clark, | Stephen |     |     |     |     |     |     |
| --------------- | --- | ------- | ----- | ------ | ------- | --- | --- | --- | --- | --- | --- |
Pfohl, Heather Cole-Lewis, Darlene Neal, et al. hikarla,SunyangFu,XunChen,ChenChen,Yuyin
2023. Towards expert-level medical question an- Zhou, Xiang Li, et al. 2023a. Biomedgpt: A uni-
sweringwithlargelanguagemodels. arXivpreprint fiedandgeneralistbiomedicalgenerativepre-trained
arXiv:2305.09617. transformer for vision, language, and multimodal
tasks. arXivpreprintarXiv:2305.17100.
| Neil C Thompson, |     | Kristjan | Greenewald, |     | Keeheon |     |     |     |     |     |     |
| ---------------- | --- | -------- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- |
Lee, and Gabriel F Manso. 2020. The compu- Sheng Zhang, Yanbo Xu, Naoto Usuyama, Jaspreet
tational limits of deep learning. arXiv preprint Bagga, Robert Tinn, Sam Preston, Rajesh Rao,
arXiv:2007.05558. Mu Wei, Naveen Valluri, Cliff Wong, et al. 2023b.

Large-scaledomain-specificpretrainingforbiomed- • Combinethemodifiedprecisionandbrevity
ical vision-language processing. arXiv preprint penaltytocomputetheBLEUscore:
arXiv:2303.00915.
(cid:32) n (cid:33)
JiaweiZhao, ZhenyuZhang, BeidiChen, Zhangyang (cid:88)
BLEU = BP·exp w logp (10)
Wang, Anima Anandkumar, and Yuandong Tian. i i
2024. Galore: Memory-efficient llm training i=1
by gradient low-rank projection. arXiv preprint
where w are the weights assigned to each
arXiv:2403.03507. i
n-gramprecision.
BaichuanZhou,YingHu,XiWeng,JunlongJia,JieLuo,
XienLiu,JiWu,andLeiHuang.2024. Tinyllava: A A.1 Datasetinformation
frameworkofsmall-scalelargemultimodalmodels.
VQA-RAD(Lauet al., 2018)contains3,515QA
arXivpreprintarXiv:2402.14289.
pairs and 315 radiology images, with questions
A Appendix
covering11categoriesandamixofclosed-ended
andopen-endedtypes. SLAKE(Liuetal.,2021)
CalculationFormulasforOpenSettingMetrics
comprises 642 radiology images and over 7,000
1. Recall
QA pairs, including segmentation masks and ob-
Recalliscalculatedusingthefollowingformula:
ject detection bounding boxes. PathVQA (He
TP etal.,2020)includes4,998pathologyimageswith
Recall = (6)
32,799 QA pairs, focusing on aspects like loca-
TP +FN
tion,shape,color,andappearance,categorizedinto
whereTP (truepositives)isthenumberofwordsin
open-ended and closed-ended types. The Pneu-
boththecandidateandthereference,andFN (false
moniaMNIST dataset focuses on pediatric chest
negatives)isthenumberofwordsinthereference
radiographsforbinaryclassificationofpneumonia
butnotinthecandidate.
versusnormal,using4,708trainingand624testim-
2. ExactMatchScore ages. OrganCMNIST (Yangetal.,2023)classifies
11 human body organs with 12,975 training and
ExactMatchScoreiscalculatedusingthefollow-
8,216testingimages. Toensureafaircomparison
ingformula:
with LLaVA-Med, we did not use the additional
Numberofmatchingwords
EMS = (7) imageclassificationtrainingdatasetswhenevalu-
Totalnumberofcandidatewords
atingVQA.Forevaluation, weusetestsetsfrom
Thisformulacalculatestheratioofthenumberof thesewidelyrecognizedmedicalVQAdatasetsand
matchingwordsinthecandidateandthereference additionallyassessclassificationperformance.
tothetotalnumberofwordsinthecandidate.
3. BLEUScore 100.0
97.5 TheBLEUScoreiscalculatedusingthefollowing
95.0
steps:
92.5
• Calculatethemodifiedprecisionp foreach
n 90.0
n-gramupton:
87.5
p =
(cid:80)
C∈Candidates
(cid:80)
ng∈C
min(Count(ng),Countmax(ng)) 85.0
n (cid:80) (cid:80) Count(ng) C∈Candidates ng∈C 82.5
(8)
80.0
where ng is the n-gram, Count(ng) is its 0 200 400 600 800 1000
Number of Training Data
count in the candidate, and Count (ng) is
max
itsmaximumcountinthereference.
• Calculatethebrevitypenalty(BP):
(cid:40)
1 ifc > r
BP = (9)
e(1−r c ) ifc ≤ r
wherecisthelengthofthecandidatesentence
andr isthelengthofthereferencesentence.
)%(
ycaruccA
Different Training Data for Router
97.6% 98.1%
96.3%
90.7%
82.3%
Figure6: Performanceofrouterpredictionswithdiffer-
entdomain-labeledtrainingdata

|     | Config    |     | StageI StageII           | StageIII |     |
| --- | --------- | --- | ------------------------ | -------- | --- |
|     | Deepspeed |     | Zero2,Zero2,Zero2offload |          |     |
Imageencoder CLIP-Large
|     | Featureselectlayer     |     | -2                    |      |     |
| --- | ---------------------- | --- | --------------------- | ---- | --- |
|     | Imageprojector         |     | 2LinearlayerswithGeLU |      |     |
|     | Epoch(sameasLLaVA-Med) |     | 1 3                   | 9    |     |
|     | Learningrate           |     | 1e-3 2e-5             | 2e-5 |     |
|     | Learningrateschedule   |     | Cosine                |      |     |
|     | Weightdecay            |     | 0.0                   |      |     |
|     | Textmaxlength          |     | 2048                  |      |     |
|     | BatchsizeperGPU        |     | 2                     |      |     |
|     | GPU                    |     | 8×3090-24G            |      |     |
|     | Precision              |     | Bf16                  |      |     |
Table11: Ourexperimentalhyperparameters
| Stage  | DataSource                    |     |     |     | SampleSize |
| ------ | ----------------------------- | --- | --- | --- | ---------- |
| Stage1 | llava_med_alignment_500k.json |     |     |     | 500K       |
| Stage2 | instruct_60k_inline_mention   |     |     |     | 60K        |
VQA:RAD-VQA,SLAKE,Path-VQA:27K
| Stage3 |     |     |     |     | 44K |
| ------ | --- | --- | --- | --- | --- |
Classification:PneumoniaMNIST,OrganCMNIST:17K
|     | Table12: SummaryofDataUtilizedAcrossTrainingStages |     |     |     |     |
| --- | -------------------------------------------------- | --- | --- | --- | --- |
Name Experts ActivatedExperts MoELayers Embedding Width Layers FFN FFNFactor Heads ActivatedParam TotalParam
| StableLM-1.6B | - - | - 100352 | 2560 32 10240 | 2 32 | 1.6B 1.6B |
| ------------- | --- | -------- | ------------- | ---- | --------- |
Med-MoE(StableLM-4x1.6B) 4 2 16 100352 2560 32 10240 2 32 2.0B 2.9B
| Phi2-2.7B | - - | - 51200 | 2560 32 10240 | 2 32 | 2.7B 2.7B |
| --------- | --- | ------- | ------------- | ---- | --------- |
Med-MoE(Phi2-4x2.7B) 4 2 16 51200 2560 32 10240 2 32 3.6B 5.3B
Table13: Comparisonofdifferentmodelsintermsofvariousparameters.

GPT4 Evaluation
Assuming you are a doctor, please rate the following three
answers about this medical picture from 0 to 10, and tell me
why.
From the perspective of a medical professional:
1.This explanation is detailed and accurate, so I'd give it an 8 out of 10. It
mentions the specific posteroanterior view, which is essential for proper
interpretation of a chest X-ray. It also lists the structures that are typically
assessed with this imaging technique, though it does not provide much detail
about what might be diagnosed with this information.
Q:What is shown in this image?
MoE-TinyMed: The image is a posteroanterior chest
X-ray, which is a common diagnostic imaging 2.I would rate this explanation a 7 out of 10. While it correctly identifies the type
technique used to visualize the structures within the of chest X-ray, it does not offer any additional details beyond what the first
chest, including the lungs, heart, and bones of the chest explanation provides. The term "surrounding structures" is somewhat vague and
and spine. could be expanded upon for clarity.
LLaVA-Med: The image is a postero-anterior chest X-
ray, which shows the patient's chest, including the 3.This response gets a 9 out of 10. It offers an extensive overview of the anatomy
heart, lungs, and surrounding structures. visible on a chest X-ray, noting the appearance of normal lung fields, the
positioning of the heart, and the visibility of the diaphragm. It goes further to
GPT4V:This image is a chest X-ray, a type of explain the clinical utility of a chest X-ray, detailing various conditions it can
radiograph that is used in medical settings to examine help diagnose. This level of detail is very informative and has significant
the chest cavity. It shows the bones of the shoulder educational value for understanding both the anatomy and the clinical reasoning
girdle and the upper arms, as well as the bony thorax behind the use of chest X-rays.
which includes the ribs, the thoracic portion of the
spine, and the sternum. It also shows the lung fields, The ratings are based on the depth and utility of the information provided. The
which should normally appear darker due to the air third response is rated highest because it provides a thorough explanation that
could enhance understanding in a clinical education context and assist in patient
content, the heart, which is usually visible in the center
care decisions.
as a denser area, and the diaphragm separating the
chest cavity from the abdominal cavity.
Figure7: Anexampleshowcasingourmethod’sabilitytoanswermedicalimagingquestionswithperformance
nearingorevensurpassingthatofLLaVA-MedunderGPT-4Vevaluation.

| Case from RAD-VQA |                                             | Case from SLAKE | Case from Path-VQA                       |
| ----------------- | ------------------------------------------- | --------------- | ---------------------------------------- |
|                   | Q:Which is smaller in this image, colon or  |                 | Q:Does typical tuberculous exudate show  |
Q:The image is taken in what plane?
| A:Axial | small bowel? |     | obvious lesion? |
| ------- | ------------ | --- | --------------- |
|         | A:Colon      |     | A:No            |
Q:Is there narrowing of the lumen of coronary due to fully
Q:Is there air outside the bowel walls? developed atheromatous plaque which has dystrophic
Q:What is the shape of spinal cord in this image?
| A:No | A:Circular |     |  calcification in its core? |
| ---- | ---------- | --- | --------------------------- |
A:No
Figure8: MoreMedicalVQAcasesfromVQA-RAD,SLAKE,andPath-VQA.ourMed-MoEgeneratesexpected
responsesformedicalimagequeries.

Q: What done external view of lacerations of
Q: What is present?
capsule done during?
A:endocrine
A:dissection
GT:cardiovascular
GT:done surgical procedure
Q: What are the hyperdense lesions
Q: What structures are involved?
noted at the edges of the aorta?
A:basal ganglia, cerebellum, cerebral cortex
A:Calcifications
GT:Caudate, putamen, left parietal
GT:Calcified atherosclerosis
Figure9: Incorrectcases: Med-VQAexamplesinOPENsettingrequiringpreciseandspecializedmedicalknowl-
edge.

Silhouette Score:0.173 Silhouette Score:0.579
(a). Embedding visualization for domain-specific tasks using Router in MoE-TinyMed-Phi2.
Figure10: ExpertsRoutingVisualizationOnPhi2
Figure11: Theactivationproportionsfortextandimageprocessinginothermodalities