#ifndef _MODELT42_H_
#define _MODELT42_H_

#include <set>
#include <vector>
#include <math.h>
#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <ros/ros.h>

#include "common_msgs_gl/SendDoubleArray.h"
#include "common_msgs_gl/SendInt.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Int32MultiArray.h"
#include "marker_tracker/ImageSpacePoseMsg.h"

class ModelT42: public Environment {
public:
    /** Creates a ModelT42 domain */
    ModelT42(Random &rand, int stateSize);
//    ModelT42(Random &rand);

    virtual ~ModelT42();

    virtual const std::vector<float> &sensation() const;
    virtual float apply(int action);

    virtual bool terminal() const;
    virtual void reset();
    virtual void reset_old();

    virtual int getNumActions();
    virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
    virtual void getMinMaxReward(float* minR, float* maxR);
    virtual void callbackImageSpacePoseMsg(marker_tracker::ImageSpacePoseMsg msg);
    virtual void callbackSliding(std_msgs::Bool msg);
    virtual void callbackStuck(std_msgs::Bool msg);
    virtual void callbackSystemState(std_msgs::Int32 msg);
    virtual void callbackGripperLoad(std_msgs::Float32MultiArray msg);
    virtual void callbackContactPointObjectLeft(std_msgs::Int32MultiArray msg);
    virtual void callbackContactPointObjectRight(std_msgs::Int32MultiArray msg);
    //virtual void callbackCarRef(std_msgs::Float64MultiArray msg);
    void setupNode();

    std::vector<std::vector<float> > getMarkerPositions();
    void setSensation(std::vector<float> newS);

    ros::Subscriber sub_vs_vel_ref_;
    ros::Subscriber sub_sd_sliding_detector;
    ros::Subscriber sub_sd_stuck_detector;
    ros::Subscriber sub_system_state;
    ros::Subscriber sub_gripper_load;
    ros::Subscriber sub_contact_point_obj_left;
    ros::Subscriber sub_contact_point_obj_right;
    ros::Publisher pub_car_ref;
    ros::ServiceClient srvclnt_set_mode_;
    ros::ServiceClient srvclnt_car_ref_;
protected:
    typedef std::pair<float,float> coord_t;
    ros::NodeHandle node_handle_;
    enum actuator_command {UP,DOWN,LEFT,RIGHT,LEFTUP,LEFTDOWN,RIGHTUP,RIGHTDOWN,STOP};
    float getEucDist(coord_t a, coord_t b) const;
private:
    coord_t goal;

    double gain_ = 0.2;
    bool reachedEnd = false;
    const bool negReward;
    const bool extraReward;
    const bool noisy;
    const bool rewardSensor;
    float goalRange = 10.;
    int maxNumRollouts = 30; // deprecated
    int numRollouts = 0; //deprecated
    const int objMarkerId = 4;
    int systemState = 0;

    actuator_command lastAction = STOP;
    bool lastSliding = false;
    bool stuck_state = false;
    int applyCount = 0;
    Random &rng;

    std::vector<float> s;

    coord_t objPos = coord_t(0,0);
    float objAngle = 0.0;
    std::vector<coord_t> markerPos{coord_t(0,0),coord_t(0,0),coord_t(0,0),coord_t(0,0),coord_t(0,0),coord_t(0,0),coord_t(0,0)};
    std::vector<float> markerAngle{0., 0., 0., 0., 0., 0., 0.};
    std::vector<coord_t> contactPointLeft{coord_t(0,0)};
    std::vector<coord_t> contactPointRight{coord_t(0,0)};

    const bool goalOption;

    /** Return the correct reward based on the current state. */
    float reward();
    std::vector<double> prepareControlVector(const actuator_command effect);
};

#endif
