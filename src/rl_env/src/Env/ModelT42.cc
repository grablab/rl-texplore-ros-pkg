#include <rl_env/ModelT42.hh>
#include <common_msgs_gl/SendBool.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>


//ModelT42::ModelT42(Random &rand, int stateSize):
//        negReward(true),
//        noisy(false),
//        extraReward(true),
//        rewardSensor(false),
//        rng(rand),
//        goalOption(false),
//        gain_(0.3),
//        s(stateSize)
//{
//    cout << "upupupupupupup" << endl;
//    setupNode();
//    reset();
//}

ModelT42::ModelT42(Random &rand, int stateSize):
        negReward(true),
        noisy(false),
        extraReward(true),
        rewardSensor(false),
        rng(rand),
        goalOption(false),
        gain_(0.3),
        s(stateSize)
{
    cout << "down down down down stateSize " << stateSize << endl;
    setupNode();
    reset();
}

ModelT42::~ModelT42() { }

void ModelT42::setupNode() {
    sub_vs_vel_ref_ = node_handle_.subscribe("/marker_tracker/image_space_pose_msg",1,&ModelT42::callbackImageSpacePoseMsg, this);
    // TODO: Think about how to fill s with contact point info as well as the regular marker observation info
    // ToDO: Finish callback for contact point
    sub_contact_point_obj_left = node_handle_.subscribe("/contact_point_detector/contact_point_object_left",1,&ModelT42::callbackContactPointObjectLeft,this);
    sub_contact_point_obj_right = node_handle_.subscribe("/contact_point_detector/contact_point_object_right",1,&ModelT42::callbackContactPointObjectRight,this);
    sub_sd_sliding_detector = node_handle_.subscribe("/evaluate_policy/evaluate_policy",6,&ModelT42::callbackSliding, this);
    sub_sd_stuck_detector = node_handle_.subscribe("/stuck_detector/stuck_detector",6,&ModelT42::callbackStuck, this);
    sub_system_state = node_handle_.subscribe("/system/state",1,&ModelT42::callbackSystemState, this);
    sub_gripper_load = node_handle_.subscribe("/gripper/load",1,&ModelT42::callbackGripperLoad, this);

    pub_car_ref = node_handle_.advertise<std_msgs::Float64MultiArray>("/visual_servoing/vel_ref_monitor", 1);

    srvclnt_set_mode_ = node_handle_.serviceClient<common_msgs_gl::SendInt>("/manipulation_manager/set_mode");
    srvclnt_set_mode_.waitForExistence(ros::Duration(5));
    srvclnt_car_ref_ = node_handle_.serviceClient<common_msgs_gl::SendDoubleArray>("/visual_servoing/vel_ref");
    srvclnt_car_ref_.waitForExistence(ros::Duration(5));



    ros::param::get("/RLAgent/goal_range", goalRange);
//    ros::param::get("/RLAgent/num_rollouts", maxNumRollouts);

    string goalStr;
    ros::param::get("/RLAgent/goal", goalStr);
    int goalx = stoi(goalStr.substr(0, goalStr.find(",")));
    goalStr.erase(0, goalStr.find(",") + 1);
    int goaly = stoi(goalStr);
    goal = coord_t(goalx, goaly);
    cout << "goal x, y: " << goalx << "," << goaly << endl;
}


void ModelT42::callbackGripperLoad(std_msgs::Float32MultiArray msg) {
    float l_load, r_load;
    l_load = msg.data[0];
    r_load = msg.data[1];
//    cout << "callbackGripperLoad is called l_load and r_load values are : " << l_load << " " << r_load << endl;
//    cout << "callbackGripperLoad is called l_load and r_load values are : " << fabs(l_load) << " " << fabs(r_load) << endl;
    if (fabs(l_load) > 900.0 || fabs(r_load) > 900.0) {
        // reset because
        cout << "load is too high" << endl;
        reachedEnd = true;
    }
}

void ModelT42::callbackContactPointObjectLeft(std_msgs::Int32MultiArray msg){
//    cout << "callback contact point object left: " << msg << endl;
//    cout << "gg: " << msg.data[0] << endl;
    contactPointLeft[0] = coord_t(msg.data[0], msg.data[1]);
}

void ModelT42::callbackContactPointObjectRight(std_msgs::Int32MultiArray msg){
//    cout << "callback contact point object right: " << msg << endl;
    contactPointRight[0] = coord_t(msg.data[0], msg.data[1]);
}


void ModelT42::callbackSystemState(std_msgs::Int32 msg) {
//    cout << "received system state: " << msg.data << endl;
    systemState = msg.data;
}


void ModelT42::callbackImageSpacePoseMsg(marker_tracker::ImageSpacePoseMsg msg) {
//    cout << "number of ids " << msg.ids.size() << ", which ids: ";
//    for(auto &a : msg.ids) {
//        cout << a << ", ";
//    }
//
//    cout << endl;
    for (unsigned i = 0; i < msg.ids.size(); i++) {
        int markerId = msg.ids[i];
        if (markerId == objMarkerId) {
            objPos = coord_t(msg.posx[i], msg.posy[i]); // coord_t(std::floor(msg.posx[i]/5), std::floor(msg.posy[i]/5)); // rounds position to 5
            objAngle = (float)msg.angles[i];
        }
        markerPos[markerId] = coord_t(msg.posx[i], msg.posy[i]); // coord_t(std::floor(msg.posx[i]/5), std::floor(msg.posy[i]/5));
        markerAngle[markerId] = (float)msg.angles[i];
    }

    std::vector<float> sensation;
    if (s.size() == 2) {
        sensation.push_back(objPos.first);
        sensation.push_back(objPos.second);
    }
    else if (s.size() == 9) {
        sensation.push_back(markerPos[0].first);
        sensation.push_back(markerPos[0].second);
        sensation.push_back(markerPos[2].first);
        sensation.push_back(markerPos[2].second);
        sensation.push_back(markerPos[4].first);
        sensation.push_back(markerPos[4].second);
        sensation.push_back(markerAngle[0]);
        sensation.push_back(markerAngle[2]);
        sensation.push_back(markerAngle[4]);
    }
    else if (s.size() == 17)  {
        for (auto &a: markerPos) { // this adds 12 values to sensation
            sensation.push_back(a.first);
            sensation.push_back(a.second);
        }
        sensation.push_back(markerAngle[0]);
        sensation.push_back(markerAngle[2]);
        sensation.push_back(markerAngle[4]);
    }
    else if (s.size() == 21)  {
        for (auto &a: markerPos) { // this adds 12 values to sensation
            sensation.push_back(a.first);
            sensation.push_back(a.second);
        }
        sensation.push_back(markerAngle[0]);
        sensation.push_back(markerAngle[2]);
        sensation.push_back(markerAngle[4]);
        sensation.push_back(contactPointLeft[0].first);
        sensation.push_back(contactPointLeft[0].second);
        sensation.push_back(contactPointRight[0].first);
        sensation.push_back(contactPointRight[0].second);
    }
//    sensation.push_back((int)lastSliding);
    setSensation(sensation);
}

void ModelT42::callbackSliding(std_msgs::Bool msg) {
    lastSliding = (bool)msg.data;
}


void ModelT42::callbackStuck(std_msgs::Bool msg) {
    stuck_state = (bool)msg.data;
    cout << "callbackStuck is called...stuck_state is " << stuck_state << endl;
}

const std::vector<float> &ModelT42::sensation() const {
  //cout << "At state " << s[0] << ", " << s[1] << endl;
  return s;
}

std::vector<double> ModelT42::prepareControlVector(const actuator_command effect) {
    switch(effect) {
        case UP:
            return {0.0,-gain_, 0.0,0.0,0.0,0.0};
        case DOWN:
            return {0.0,gain_,0.0,0.0,0.0,0.0};
        case LEFT:
            return {gain_, 0.0, 0.0, 0.0, 0.0, 0.0};
        case RIGHT:
            return {-gain_,0.0,.0,0.0,0.0,0.0};
        case LEFTUP:
            return {gain_,-gain_,0.0,0.0,0.0,0.0};
        case LEFTDOWN:
            return {gain_,gain_,0.0,0.0,0.0,0.0};
        case RIGHTUP:
            return {-gain_,-gain_,0.0,0.0,0.0,0.0};
        case RIGHTDOWN:
            return {-gain_,gain_,0.0,0.0,0.0,0.0};
        case STOP:
            return {0.0,0.0,0.0,0.0,0.0,0.0};
        default:
            cout << "WARNING: control action not recognized, defaulting to STOP" << endl;
            return {0.0,0.0,0.0,0.0,0.0,0.0};
    }
}

float ModelT42::apply(int action) {
    // I should just have apply_count and then ignore any stuck_state when apply_count < 10 or something.
    // because apply command should be sent by 5 Hz or something right? So if I wait for like 3,4 seconds,
    // which ammounts to 15 to 20 apply_counts, the T42 should be fine.
    // Make sure that apply_count is reset in the reset function too tho.
    applyCount += 1;
    cout << "ModelT42::applyCount is " << applyCount << endl;

    if (stuck_state) {
        cout << "stuck_state is 1...reseting the env.............." << endl;
        reset();
        stuck_state = false;
        return 0;
    }
    if (reachedEnd || applyCount > 40) {
        reset();
        return 0;
    }

    const actuator_command effect = static_cast<actuator_command>(action);
    lastAction = effect;
    //cout << "Taking action " << effect << endl;
    std::vector<double> car_ref;
    car_ref = prepareControlVector(effect);

    common_msgs_gl::SendDoubleArray srv_out;
    srv_out.request.data = car_ref;
    if(!srvclnt_car_ref_.call(srv_out)){
        throw std::runtime_error("[ModelT42-RLEnv] Failed sending to send action");
    }
    // send car_ref as message too since the robot gui expects it as message
    std_msgs::Float64MultiArray msg;
    msg.data = car_ref;
    pub_car_ref.publish(msg);
    std::cout<<"Applying command!"<<std::endl;

    float rew = lastSliding; // reward(); // I think I can just do float rew = lastSliding
    return rew;
}


float ModelT42::reward() {
    if (extraReward){
        if (terminal()) {
            reachedEnd = true;
            return (float) 0;
        }
        else {
            float dist = (float)-.1 * getEucDist(objPos, goal);
            return dist;
        }
    }

    if (negReward){
    // normally -1 and 0 on goal
        if (terminal()){
            reachedEnd = true;
            return (float)0;
        }
        else
            return (float)-1;
    }
    else {
    // or we could do 0 and 1 on goal
        if (terminal()){
            reachedEnd = true;
            return (float)1;
        }
        else
            return (float)0;
    }
}

float ModelT42::getEucDist(coord_t a, coord_t b) const{
    int firstDiff = a.first - b.first;
    int secondDiff = a.second - b.second;
    return (float)sqrt(pow(firstDiff, 2) + pow(secondDiff, 2));
}

bool ModelT42::terminal() const {
  // current position equal to goal??
    float dist = getEucDist(objPos, goal);
    return dist < goalRange;
}

void ModelT42::reset() {
    // publish signal to object resetter
    reachedEnd = false;
    applyCount = 0;

    cout << "successful env reset" << endl;
}

void ModelT42::reset_old() {
    // reset without object resetter node
    // make sliding manager run getReady
    reachedEnd = false;
    applyCount = 0;

    bool resetFlag = false;
    cout << "Printing resetFlag: " << resetFlag << endl;
    ros::param::get("/RLAgent/reset", resetFlag);
    cout << "Printing resetFlag: " << resetFlag << endl;
    ros::param::set("/RLAgent/reset", resetFlag);
    numRollouts += 1;

    common_msgs_gl::SendInt mode;
    mode.request.data = 0;
    systemState = 4; // what's the meaning of this assignment? Isn't systemState automatically updated every time ModelT42::callbackSystemState is called?

    cout << "debugging 1" << endl;
    // send get_ready command to /system/mode node
    if (!srvclnt_set_mode_.call(mode)) {
        cout << "debugging 2" << endl;
        throw std::runtime_error("[ModelT43-RLEnv] Cannot send reset mode");
    }

    if (numRollouts > maxNumRollouts) {
        cout << "reached maximum number of rollouts: " << maxNumRollouts << endl;
        ros::shutdown();
        exit(0);
    }

    ros::Rate r = ros::Rate(10);
    int counter = 0; // counter waits for up to 10 seconds

    // wait for system mode to be "ready"
    while (systemState != 3 and counter < 600) {
        ros::spinOnce();
        r.sleep();
        counter++;
    }
    if (systemState != 3) {
        throw std::runtime_error("[ModelT42-RLEnv] Failed to get ready");
    }

    cout << "waiting for /RLAgent/reset param set to true" << endl;
    while(!resetFlag) {
        ros::param::get("/RLAgent/reset", resetFlag);
        ros::Duration(2).sleep();
    }

    // send start command to /system/mode service
    mode.request.data = 1;
    if (!srvclnt_set_mode_.call(mode)) {
        throw std::runtime_error("[ModelT42-RLEnv] Cannot send start mode");
    }
    counter = 0; // counter waits for up to 10 seconds
    cout << "successful get ready" << endl;
    // wait for system mode to be "vel_control"
    while (systemState != 2 and counter < 100) {
        ros::spinOnce();
        r.sleep();
        counter++;
    }
    if (systemState != 2) {
        throw std::runtime_error("[ModelT42-RLEnv] Failed to start running+vel_control");
    }

    cout << "successful env reset" << endl;
}

int ModelT42::getNumActions(){
  return 9;
}

/** For special use to test true transitions */
void ModelT42::setSensation(std::vector<float> newS){
  if (s.size() != newS.size()){
    cerr << "Error in sensation sizes" << endl;
  }

  for (unsigned i = 0; i < newS.size(); i++){
    s[i] = newS[i];
  }
}


void ModelT42::getMinMaxFeatures(std::vector<float> *minFeat,
                                  std::vector<float> *maxFeat){

    minFeat->resize(s.size(), 0.0);
    maxFeat->resize(s.size(), 0.0);
    int i;
    for (i = 0; i + 2 < s.size(); i++){
        (*minFeat)[i] = 0;
        if (i % 2 == 0) {
            (*maxFeat)[i] = (float)std::floor(1024/5);
        }
        else {
            (*maxFeat)[i] = (float)std::floor(576/5);
        }
    }
    (*maxFeat)[i] = 8;
    (*maxFeat)[i+1] = 1;
}

void ModelT42::getMinMaxReward(float *minR,
                               float *maxR){

  if (extraReward){
    *minR = -5.0;
    *maxR = 0.0;
  }
  else if (negReward){
    *minR = -1.0;
    *maxR = 0.0;
  }else{
    *minR = 0.0;
    *maxR = 1.0;
  }

}
