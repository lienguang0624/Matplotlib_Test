import  matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 在不同的窗口显示不同的图像
plt.figure()
# 绘制线性方程
x = np.linspace(-3,3,100)
y1 = 2*x + 1
y2 = x**2

x0 = 1
y0 = 2*x0 + 1
# 绘制坐标轴
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
# 标注指定点
plt.scatter(x0,y0,s = 10,color = 'b')
# 画出函数曲线
plt.plot(x,y1)
plt.plot([x0,x0],[y0,0],'k--',lw = 2.5)
plt.annotate(r'$2x+1=%s$'%y0, xy=(x0, y0), xycoords = 'data',xytext = (+30,-30),textcoords = 'offset points',
             fontsize = 16,arrowprops = dict(arrowstyle = '->',connectionstyle = 'arc3,rad=.2'))
# 指定位置添加自定义字符串
plt.text(-3.7,3,r'$love\ fish\ forever$',fontdict = {'size':16,'color':'r'})
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\绘制线性方程.png')




# 在同一窗口显示不同图像，并改变线的参数
plt.figure()
l1, = plt.plot(x,y2,label = 'up')
l2, = plt.plot(x,y1,color = 'red',linewidth = 1.0,linestyle = '--',label = 'down')
# 设置x，y轴取值范围
plt.xlim((-1,2))
plt.ylim((-2,3))
# 记录x，y轴是什么东西
plt.xlabel('I am x')
plt.ylabel('I am y')
# 坐标轴替换
new_ticks = np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3.],[r'$really\ bad$',r'$bad\ \alpha$',r'$normal$',r'$good$',r'$really\ good$'])
# 移动坐标轴
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
# 制作图例
plt.legend(handles=[l1,l2],labels=['aaa','bbb'],loc='best')
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\坐标轴标注移动坐标轴绘制图例.png')




# 对比度
x = np.linspace(-3,3,50)
y = 0.1*x
plt.figure()
plt.plot(x,y,linewidth = 10,zorder = 1)
plt.ylim(-2,2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))#坐标上的数值被挡住了
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_zorder(100)
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.7))
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\更改线条对比度防止遮挡.png')

# 散点图
plt.figure()
n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X)
plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.xticks(())
plt.yticks(())
plt.scatter(X,Y,s = 75,c = T)
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\散点图绘制.png')
# 柱状图
plt.figure()
n = 12
X = np.arange(n)
Y1 = (1-X/float(n))*np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n))*np.random.uniform(0.5,1.0,n)
plt.bar(X,+Y1,facecolor = '#9999ff',edgecolor = 'white')
plt.bar(X,-Y2)
for x,y in zip(X,Y1):
    plt.text(x ,y + 0.05,'%.2f'%y,ha = 'center',va = 'bottom')
for x,y in zip(X,Y2):
    plt.text(x ,-y - 0.05,'-%.2f'%y,ha = 'center',va = 'top')
plt.xlim(-0.5,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\柱状图绘制.png')
# 等高线图
plt.figure()
def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.contourf(X,Y,f(X,Y),8,alpha = 0.7,cmap = plt.cm.hot)
C = plt.contour(X,Y,f(X,Y),8,colors = 'black',linewidths = .5)
plt.clabel(C,inline = True,fontsize = 10)
plt.xticks(())
plt.yticks(())
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\等高线图绘制.png')
# 绘制2D图像
plt.figure()
a = np.array([0.111111111111,0.2222222222,0.333333333,
              0.444444444444,0.5555555555,0.666666666,
              0.777777777777,0.8888888888,0.999999999]).reshape(3,3)
plt.imshow(a,interpolation='nearest',cmap = 'bone',origin = 'upper')
plt.colorbar(shrink = 0.9)
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\2D图像绘制.png')
# 绘制3D图像
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4,4,0.25)
Y = np.arange(-4,4,0.25)
X,Y = np.meshgrid(X,Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
#3D图
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap = plt.get_cmap('rainbow'))
#在z轴绘制等高线图
ax.contourf(X,Y,Z,zdir='z',offset = -2,cmap = plt.get_cmap('rainbow'))
ax.set_zlim(-2,2)
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\3D图像绘制.png')
#多图合一显示
plt.figure()
plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
plt.subplot(2,3,4)
plt.plot([0,1],[0,2])
plt.subplot(2,3,5)
plt.plot([0,1],[0,3])
plt.subplot(2,3,6)
plt.plot([0,1],[0,4])
plt.savefig('C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\TensorFlow\\Matplotlib_test\\多图合一.png')

plt.show()