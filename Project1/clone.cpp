#include "clone.h"
#include "poisson_solver.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>

//计算overlap重叠区域
bool findOverlap(cv::InputArray background,cv::InputArray foreground,int offsetX, int offsetY,cv::Rect &rBackground,cv::Rect &rForeground)
{
    cv::Mat bg = background.getMat();
    cv::Mat fg = foreground.getMat();

    rBackground = cv::Rect(0, 0, bg.cols, bg.rows) & 
                  cv::Rect(offsetX, offsetY, fg.cols, fg.rows);

    // Compensate for negative offsets. If offset < 0, offset in foreground is positive.
    rForeground = cv::Rect(std::max<int>(-offsetX, 0), 
                           std::max<int>(-offsetY, 0), 
                           rBackground.width, 
                           rBackground.height);

    return rForeground.area() > 0;
    
}
//计算复合梯度场，定义kernel，调用filter2D扫描一遍图像函数
void computeMixedGradientVectorField(cv::InputArray background,cv::InputArray foreground,cv::OutputArray vx_,cv::OutputArray vy_)
{
    cv::Mat bg = background.getMat();
    cv::Mat fg = foreground.getMat();
    
    const int channels = bg.channels();
    
    vx_.create(bg.size(), CV_MAKETYPE(CV_32F, channels));
    vy_.create(bg.size(), CV_MAKETYPE(CV_32F, channels));
    
    cv::Mat vx = vx_.getMat();
    cv::Mat vy = vy_.getMat();
    //定义两个方向上的卷积核
    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    //卷积得到梯度场
    cv::Mat vxf, vyf, vxb, vyb;
    cv::filter2D(fg, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(fg, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    
    //给所有点编号，根据0或1将梯度数据编起来
    for(int id = 0; id <= (vx.rows * vx.cols * channels - channels); ++id)
    {
		//Vec2f就是两维的float类型数组
        const cv::Vec2f g[2] = {
            cv::Vec2f(vxf.ptr<float>()[id], vyf.ptr<float>()[id]),
            cv::Vec2f(vxb.ptr<float>()[id], vyb.ptr<float>()[id])
        };
        
        int which = (g[0].dot(g[0]) > g[1].dot(g[1])) ? 0 : 1;
        
        vx.ptr<float>()[id] = g[which][0];
        vy.ptr<float>()[id] = g[which][1];
    }
}
//添加权重后的梯度场，可以调节前后景的比例，这部分可以手动调节
void computeWeightedGradientVectorField(cv::InputArray background,cv::InputArray foreground,cv::OutputArray vx,cv::OutputArray vy,float weightForeground)
{
    
    cv::Mat bg = background.getMat();
    cv::Mat fg = foreground.getMat();
    
    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    
    cv::Mat vxf, vyf, vxb, vyb;
    cv::filter2D(fg, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(fg, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    //根据权重分配梯度
    cv::addWeighted(vxf, weightForeground, vxb, 1.f - weightForeground, 0, vx);
    cv::addWeighted(vyf, weightForeground, vyb, 1.f - weightForeground, 0, vy);
}
//考虑将计算梯度的范围从全域缩小到ROI区域，同时使用较快的泊松方程（线性大方程组）解法提高效率
void seamlessCloneNaive(cv::InputArray background,
                 cv::InputArray foreground,
                 cv::InputArray foregroundMask,
                 int offsetX,
                 int offsetY,
                 cv::OutputArray destination,
                 CloneType type)
{
        //复制背景图片作为底版，之后在对应区域的修改都只需要覆盖这上面的区域就行了
        background.getMat().copyTo(destination);
        
        //检查重叠区域，如果并没有就立即返回（往往此时的偏置输入是有问题的）
        cv::Rect rbg, rfg;
        if (!findOverlap(background, foreground, offsetX, offsetY, rbg, rfg))
            return;
        
        //计算指导向量场
        cv::Mat vx, vy;
        computeWeightedGradientVectorField(background.getMat()(rbg),foreground.getMat()(rfg), vx, vy, 1.f);
        
        //得到指导向量场基础上再次求梯度卷积得到需要的散度
        cv::Mat vxx, vyy;
        cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
        cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
        cv::filter2D(vx, vxx, CV_32F, kernelx);
        cv::filter2D(vy, vyy, CV_32F, kernely);
        //总卷积结果
        cv::Mat f = vxx + vyy;
                
        cv::Mat boundaryMask(rfg.size(), CV_8UC1);      
        cv::threshold(foregroundMask.getMat()(rfg), boundaryMask, UNKNOWN, DIRICHLET_BD, cv::THRESH_BINARY_INV);
        cv::rectangle(boundaryMask, cv::Rect(0, 0, boundaryMask.cols, boundaryMask.rows), DIRICHLET_BD, 1);

        cv::Mat boundaryValues(rfg.size(), CV_MAKETYPE(CV_32F, background.channels()));
        background.getMat()(rbg).convertTo(boundaryValues, CV_32F);
        //解泊松方程
        cv::Mat result;
        solvePoissonEquations(f,
                              boundaryMask,
                              boundaryValues,
                              result);
        
        //返回结果，返回到目标图像的对应Mat区域
        result.convertTo(destination.getMat()(rbg), CV_8U);
}
//还有没有可能进一步降低需要精确计算的方程数量？用插值方法，那就要找到一部分有代表性的点，或者是找出对比较大的区块，于是使用四叉树快速解
void seamlessClone(cv::InputArray background,
                 cv::InputArray foreground,
                 cv::InputArray foregroundMask,
                 int offsetX,
                 int offsetY,
                 cv::OutputArray destination,
                 CloneType type)
{
  
    //兴趣区域
    background.getMat().copyTo(destination);
    
    //重合区域
    cv::Rect rbg, rfg;
	//特判
    if (!findOverlap(background, foreground, offsetX, offsetY, rbg, rfg))
        return;
    //划分区域
    cv::Mat fore, back;
    cv::Mat lap = (cv::Mat_<float>(3, 3) << 0.0, -1, 0.0, -1, 4, -1, 0.0, -1, 0.0);        
    cv::filter2D(foreground.getMat()(rfg), fore, CV_32F, lap, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(background.getMat()(rbg), back, CV_32F, lap, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::Mat f = fore;

    cv::Mat boundaryMask(rfg.size(), CV_8UC1);      
    cv::threshold(foregroundMask.getMat()(rfg), boundaryMask, UNKNOWN, DIRICHLET_BD, cv::THRESH_BINARY_INV);
    cv::rectangle(boundaryMask, cv::Rect(0, 0, boundaryMask.cols, boundaryMask.rows), DIRICHLET_BD, 1);

    cv::Mat boundaryValues(rfg.size(), CV_MAKETYPE(CV_32F, background.channels()));
    background.getMat()(rbg).convertTo(boundaryValues, CV_32F);
    
    cv::Mat foreValues(rfg.size(), CV_MAKETYPE(CV_32F, foreground.channels()));
    foreground.getMat()(rfg).convertTo(foreValues, CV_32F);

   //解泊松方程；关键改进：四分法构建四叉树确定求解范围
    cv::Mat result;
	//时间对比
    /*
    solvePoissonEquations(f,
                          boundaryMask,
                          boundaryValues,
                          result);
    */
    solvePoissonEquationsFast(foreValues,
                          boundaryMask,
                          boundaryValues,
                          result);
    
    //最终结果
    result.convertTo(destination.getMat()(rbg), CV_8U);
    
}
    
    