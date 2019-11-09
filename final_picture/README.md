

# Improving-Reliability-for-Federated-Learning-in-Mobile-Edge-Networks

## Figure - main - zipf

* our main graph: average error rate - number of devices

* device error rate: zipf distribution with parameter 2, after getting the distribution, we divide it by 100

* data size: zipf distribution with parameter 2

* data ratio: 0.5

* To be more specific: 

  > ```python
  > small,相对于greedy提升: 0.4924601179876383
  > small,相对于SLSQP提升: 0.21309760822744192
  > medium,相对于greedy提升: 0.3892033319239571
  > medium,相对于SLSQP提升: 0.13570298910442993
  > large,相对于greedy提升: 0.3000516497088584
  > large,相对于SLSQP提升: 0.19056149281682314
  > 全局,相对于greedy最小提升： 0.15927065097312595
  > 全局,相对于greedy最大提升： 0.7265050439693782
  > 全局,相对于greedy平均提升： 0.38896538144376736
  > 全局,相对于SLSQP最小提升： -0.2140927362917622
  > 全局,相对于SLSQP最大提升： 0.39576960107326214
  > 全局,相对于SLSQP平均提升： 0.1457862136147851
  > ```

## Figure - ratio - zipf

* average error rate performance - data ratio

* device error rate: zipf distribution with parameter 2, after getting the distribution, we divide it by 100

* data size: zipf distribution with parameter 2

* online device numbers: 25

* To be more specific: 

  > ```python
  > small,相对于greedy提升: 0.4235255656854579
  > small,相对于SLSQP提升: 0.2561197749894512
  > medium,相对于greedy提升: 0.4540778865118285
  > medium,相对于SLSQP提升: 0.1632340758167984
  > large,相对于greedy提升: 0.28726265423512476
  > large,相对于SLSQP提升: 0.15507535804171002
  > 全局,相对于greedy最小提升： -0.009210402111178045
  > 全局,相对于greedy最大提升： 0.8099652865113203
  > 全局,相对于greedy平均提升： 0.37818609735323583
  > 全局,相对于SLSQP最小提升： -0.18115725016794756
  > 全局,相对于SLSQP最大提升： 0.5033772946525363
  > 全局,相对于SLSQP平均提升： 0.14579754068502154
  > ```

## Figure - ratio - f

- average error rate performance - data ratio

- device error rate: F distribution with parameter (1,1), after getting the distribution, we divide it by 5

- data size: F distribution with parameter (1,1)

- online device numbers: 25

- To be more specific: 

  > ```python
  > small,相对于greedy提升: 0.4557158206215211
  > small,相对于SLSQP提升: 0.3042362559022756
  > medium,相对于greedy提升: 0.3531324977932644
  > medium,相对于SLSQP提升: 0.03842385915661154
  > large,相对于greedy提升: 0.16787394885166534
  > large,相对于SLSQP提升: 0.17094966576014922
  > 全局,相对于greedy最小提升： -0.06737339897489557
  > 全局,相对于greedy最大提升： 0.7292001275903853
  > 全局,相对于greedy平均提升： 0.30980407506510177
  > 全局,相对于SLSQP最小提升： -0.30664719957326914
  > 全局,相对于SLSQP最大提升： 2.0115336285056085
  > 全局,相对于SLSQP平均提升： 0.23965262866008455
  > ```

## Figure - ratio - uniform

* average error rate performance - data ratio

* device error rate: uniform distribution on [0,0.2]

* data size: uniform distribution on [0,1]

* online device numbers: 25

* To be more specific: 

  > ```
  > small,相对于greedy提升: 0.3995817834144879
  > small,相对于SLSQP提升: 0.39861529241831656
  > medium,相对于greedy提升: 0.09101018896032653
  > medium,相对于SLSQP提升: 0.1889016229069118
  > large,相对于greedy提升: -0.011293527240038539
  > large,相对于SLSQP提升: 0.019619432205150507
  > 全局,相对于greedy最小提升： -0.04901354118315896
  > 全局,相对于greedy最大提升： 0.5304155300629732
  > 全局,相对于greedy平均提升： 0.14266018081642892
  > 全局,相对于SLSQP最小提升： -0.04957348124029826
  > 全局,相对于SLSQP最大提升： 0.9496724439966612
  > 全局,相对于SLSQP平均提升： 0.21186058079253586
  > ```

## Figure - bar - both distributions are various

* best error rate performance upon different distributions

* For uniform distribution, the device error ranges are:

  * [0,1] / 5
  * [1,2] / 5
  * [2,3] / 5

  and the data size ranges are:

  * [0,1]
  * [1,2]
  * [2,3]

* For F distribution, the device error ranges are:

    - (1,1) / 5
    - (8,3) / 5
    - (20,20) / 5

   and the data size ranges are:

   - (1,1)
   - (8,3)
   - (20,20)

* For zipf distribution, the device error ranges are:

  - 1.5 / 100
  - 2 / 100
  - 3 / 100

  and the data size ranges are:

  - 1.5
  - 2
  - 3

* data ratio: 0.5

* online device number: 25

* To be more specific: 

  > ```python
  > uniform分布,small,相对于greedy提升: 0.43876329007451603
  > uniform分布,small,相对于SLSQP提升: 0.4381275804290543
  > uniform分布,medium,相对于greedy提升: 0.163567393118934
  > uniform分布,medium,相对于SLSQP提升: 0.42395981459081417
  > uniform分布,large,相对于greedy提升: 0.246786386060319
  > uniform分布,large,相对于SLSQP提升: 0.11776952907384786
  > f分布,small,相对于greedy提升: 0.31088498816749305
  > f分布,small,相对于SLSQP提升: 0.5055932633845275
  > f分布,medium,相对于greedy提升: 0.5621499089885922
  > f分布,medium,相对于SLSQP提升: 0.6721712077881525
  > f分布,large,相对于greedy提升: 0.7789260926492565
  > f分布,large,相对于SLSQP提升: 0.8417459536941804
  > zipf分布,small,相对于greedy提升: 0.9434280176926076
  > zipf分布,small,相对于SLSQP提升: 0.8122455206093158
  > zipf分布,medium,相对于greedy提升: 0.5263154798640055
  > zipf分布,medium,相对于SLSQP提升: 0.5263154798640055
  > zipf分布,large,相对于greedy提升: 0.4808435675699952
  > zipf分布,large,相对于SLSQP提升: 0.5604893085341159
  > ```

## Figure - bar - only device error distributions are various

* best error rate performance upon different distributions in device error rate but fix data distribution on (1,1) F distribution

* the setting is the same as figure 5 except that data distribution is fixed

  > ```python
  > uniform分布,small,相对于greedy提升: 0.5755465529129367
  > uniform分布,small,相对于SLSQP提升: 0.7665102839860015
  > uniform分布,medium,相对于greedy提升: 0.299096818222461
  > uniform分布,medium,相对于SLSQP提升: 0.5320995997114417
  > uniform分布,large,相对于greedy提升: 0.0976878089679025
  > uniform分布,large,相对于SLSQP提升: 0.10301065472833051
  > f分布,small,相对于greedy提升: 0.5229756838232974
  > f分布,small,相对于SLSQP提升: 0.5229756838232974
  > f分布,medium,相对于greedy提升: 0.3632245007158276
  > f分布,medium,相对于SLSQP提升: 0.3632245007158276
  > f分布,large,相对于greedy提升: 0.4135137935871781
  > f分布,large,相对于SLSQP提升: 0.0
  > zipf分布,small,相对于greedy提升: 0.9448415531793901
  > zipf分布,small,相对于SLSQP提升: 0.9192489653806374
  > zipf分布,medium,相对于greedy提升: 0.6923721474472584
  > zipf分布,medium,相对于SLSQP提升: 0.6923721474472584
  > zipf分布,large,相对于greedy提升: 0.5446353510058206
  > zipf分布,large,相对于SLSQP提升: 0.0
  > ```

## Table

* time used by each methods

* the slowest one is SLSQP, the middle one is ranRFL, and the fastest one is GDS

* through import raw data to csv in Excel, you can see it clearly

* To be more specific: 这个表描述了算法运行时间随侯选设备数的变化关系，我们的算法在时间性能上优于SLSQP算法，劣于贪心算法，但随候选设备数增大，我们的算法运行时间仍在合理、可控的范围内

  > raw data:
  >
  > ```python
  > [0.014059209823608398, 0.01719050407409668, 0.023437690734863282, 0.024990296363830565, 0.02811589241027832, 0.029690980911254883, 0.026556968688964844, 0.032808351516723636, 0.031247878074645997, 0.03905167579650879, 0.037485289573669436, 0.031241798400878908, 0.05000548362731934, 0.0390739917755127, 0.04999217987060547, 0.05156033039093018, 0.048441100120544436, 0.05156407356262207, 0.05000307559967041, 0.05468838214874268, 0.054689621925354, 0.05155661106109619, 0.05468108654022217, 0.05780477523803711, 0.0640561819076538, 0.05624854564666748, 0.06874308586120606, 0.07186903953552246, 0.08750009536743164, 0.06718704700469971, 0.07811534404754639, 0.0798715353012085, 0.09139437675476074, 0.11626694202423096, 0.08905549049377441, 0.12340848445892334, 0.13746826648712157, 0.12965703010559082][0.0, 0.0, 0.0, 0.0015620708465576172, 0.0, 0.0, 0.001558399200439453, 0.0, 0.0, 0.0, 0.0, 0.001562356948852539, 0.0015625476837158204, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001565861701965332, 0.0, 0.0, 0.0, 0.0015621423721313477, 0.0, 0.0015622377395629883, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001565241813659668, 0.0031272411346435548, 0.0, 0.0031244516372680663, 0.001562190055847168][0.017183780670166016, 0.024987053871154786, 0.03279924392700195, 0.034370827674865725, 0.0437424898147583, 0.05936057567596435, 0.04842586517333984, 0.08278958797454834, 0.06716670989990234, 0.11091492176055909, 0.09841911792755127, 0.14684040546417237, 0.1108975887298584, 0.16712429523468017, 0.1999505043029785, 0.19213566780090333, 0.19993822574615477, 0.18743271827697755, 0.2108734130859375, 0.2639873266220093, 0.3030438184738159, 0.23587255477905272, 0.26711876392364503, 0.2530567407608032, 0.2963627576828003, 0.3577136993408203, 0.29209945201873777, 0.4545668840408325, 0.38426618576049804, 0.443630313873291, 0.3967693328857422, 0.5453415155410767, 0.5831639528274536, 0.5726602792739868, 0.5186112403869629, 0.5045668363571167, 0.6061081647872925, 0.7763815879821777]
  > ```

## Conclusion

* 做了这么多对比，我们的算法就一个字——好！
* 在返回解的优越性方面，我们的算法可依赖性很高；在时间性能方面，我们的算法时间性能居中，在可以接受的范围内
* 综合来看，还是一个字——好！
