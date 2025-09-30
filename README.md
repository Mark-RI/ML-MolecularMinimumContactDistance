# ML-MolecularMinimumContactDistance
Machine Learning Algorithm. Find the minimum distance between two disks in three spatial dimensions. Ultimate aim is to find the kihara energy between two molecules.

![Description of the image](https://github.com/Mark-RI/ML-MolecularMinimumContactDistance/raw/main/assets/MinimumDistanceCircles3D.png)

Definition of the parametric equation for a circle. 
$Point (\theta, r)=center+(rsin(\theta)v+rcos(\theta)u)$  $r \in (0, r]$ 

v and u are vectors that are orthoganol to each other and the normal (orientation vector). v and u are unit vectors that are then scaled by r. Center is a vector that shifts the position of the circles.  By varying r and $\theta$ why can find any point on the circle. r is constrained to be between 0 and r.

You can use torch.clamp to constrain r.

By implementing a machine learning method you can find the minimum distance between the circles. I found that starting with r = r max works really well to avoid minimums. The inital value of $\theta$ can be random.

# Energy Minimizer Machine Learning Algorithm  

Find the optimal configuration of benezene molecules in liquid and gas.

Optimal Gas configuration.
![Description of the image](https://github.com/Mark-RI/ML-MolecularMinimumContactDistance/raw/main/assets/GasMinimumEnergyTransformation.png)


Optimal Liquid configuration.
![Description of the image](https://github.com/Mark-RI/ML-MolecularMinimumContactDistance/raw/main/assets/LiquidMinimumEnergyTransformation.png)