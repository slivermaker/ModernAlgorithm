### anaconda
https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
更换源


### conda操作
> https://gengzhige-essay.readthedocs.io/docs/01%20%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/1-2%20conda%E7%9A%84%E4%BD%BF%E7%94%A8.html

```
1. conda --version #查看conda版本，验证是否安装
2. conda update conda #更新至最新版本，也会更新其它相关包
3. conda update --all #更新所有包
4. conda update package_name #更新指定的包
5. conda create -n env_name package_name #创建名为env_name的新环境，并在该环境下安装名为package_name 的包，可以指定新环境的版本号，例如：conda create -n python2 python=python2.7 numpy pandas，创建了python2环境，python版本为2.7，同时还安装了numpy pandas包
6. source activate env_name #切换至env_name环境
7. source deactivate #退出环境
8. conda info -e #显示所有已经创建的环境
9. conda create --name new_env_name --clone old_env_name #复制old_env_name为new_env_name
10. conda remove --name env_name –all #删除环境
11. conda list #查看所有已经安装的包
12. conda install package_name #在当前环境中安装包
13. conda install --name env_name package_name #在指定环境中安装包
14. conda remove -- name env_name package #删除指定环境中的包
15. conda remove package #删除当前环境中的包
16. conda env remove -n env_name #采用第10条的方法删除环境失败时，可采用这种方法
```


## 用vscode打开服务器时需要密码

> jupyter notebook list
输入命令后复制带token的命令即可