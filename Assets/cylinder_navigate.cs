/*
Requirements:

    - Install `ml-agent` through the Package Manager

    - Add this script to the agent (camera)
        + Set the target.s

    - Add a `Decision Requester` to the agent
        + Set the decision period

    - Add a `Behavior Parameter`
        + Set up the name: `CameraControl`
        + Set the observation space:
            * Space size = x
            * Stacked vector = 1
        + Set the action space:
            * Continuous actions = 0
            * Discrete actions = 2 (L, R)
*/


using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class cylinder_navigate : Agent
{

    private Camera main;

    public Vector3 starting;

    private float theta, deltaSpeed, distance, max_distance;
    private const float speedNorm = 10.0f;
    private float[] occupancy_map;
    private int Maxdetectors;
    private float detect_dist;

    private float rewardCollision;

    public int recordedStep=0;
    EnvironmentParameters resetParams;

    public float timeScaleValue=1.0f;

    public Transform[] vTarget;
    private int index, action;

    Vector3 direction, target;

    public Color color;
    public float Width;
    public Material Mat;
    public GameObject PrefabModel;
    private GameObject newModel;
    public float ModelSize;
    LineRenderer L;

    LineRenderer Add_line(Vector3 start, Vector3 end, Material M) {
        GameObject child = new GameObject();
        LineRenderer line = child.AddComponent<LineRenderer>();

        line.enabled = true;
        line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        line.receiveShadows = false;
        line.material = M;
        line.startColor = color;
        line.endColor = color;
        line.startWidth = Width;
        line.endWidth = Width;
        line.positionCount =3;

        line.SetPosition(0, start);
        line.SetPosition(1, end);
        line.SetPosition(2, end);

        return line;
    }

    void change_line(LineRenderer line, Vector3 start, Vector3 end) {
        line.SetPosition(0, start);
        line.SetPosition(1, end);
        line.SetPosition(2, end);
    }

    public override void Initialize()
    {
        // Initialize env
        resetParams = Academy.Instance.EnvironmentParameters;
        main = gameObject.GetComponent<Camera>();

        deltaSpeed = 0.2f / speedNorm;
        Maxdetectors = 60;
        detect_dist = 5f;
        max_distance = 10;

        Time.timeScale=timeScaleValue;
        Application.runInBackground=true;
        L = Add_line(starting, starting, Mat);

        newModel = Instantiate(PrefabModel, transform.position, transform.rotation);
                newModel.GetComponentsInChildren<Renderer>().ToList().ForEach(M =>
                {
                    M.sharedMaterial = Mat;
                    M.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    M.receiveShadows = false;
                });
                newModel.transform.localScale = Vector3.one * ModelSize;

        // Display.displays[0].Deactivate();
    }

    public override void OnEpisodeBegin()
    {
        // Initialize the camera and cube position
        this.initializeScene();
        recordedStep=0;
    }

    public override async void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(distance);
        sensor.AddObservation(Vector3.Cross(direction, Vector3.Normalize(target-transform.position))[1]);
        sensor.AddObservation(Vector3.Dot(direction, Vector3.Normalize(target-transform.position)));
        sensor.AddObservation(occupancy_map);
    }

    public override async void OnActionReceived(ActionBuffers actionBuffers)
    {
        recordedStep+=1;

        // Update camera angle according to the received action
        // Discrete agent
        action = actionBuffers.DiscreteActions[0];
        rewardCollision = 0;

        float lst_dist = distance;


        theta += (action-3) * deltaSpeed;
        
        direction = new Vector3(Mathf.Cos(theta), 0, Mathf.Sin(theta));
        transform.LookAt(transform.position+direction);

        transform.position += 0.5f * direction / speedNorm;

        // distance
        distance = (target-transform.position).magnitude / max_distance;
        rewardCollision += (lst_dist - distance) * speedNorm * max_distance;

        newModel.transform.position = transform.position;
        newModel.transform.rotation = transform.rotation;

        // angle
        // rewardCollision += Vector3.Dot(direction, Vector3.Normalize(target-transform.position));

        get_occupancy_map();

        int flag = 1;

        for (int i = 0; i < Maxdetectors; ++i)
            if (occupancy_map[i] * detect_dist < 0.1f) {
              rewardCollision -= 1f;
              flag = 0;
              Debug.Log("Crash");
              break;
            }
        
        if ((transform.position-target).magnitude < 0.5f) {
            target = random_position();
            target[1] = transform.position[1];
            distance = (target-transform.position).magnitude / max_distance;
            rewardCollision += 5; //Vector3.Dot(direction, Vector3.Normalize(target-transform.position));
            Debug.Log("arrive");
        }
        SetReward(rewardCollision);
        change_line(L, transform.position, target);

        if (flag == 0)
            EndEpisode();
    }

    private void get_occupancy_map() {
        occupancy_map = new float[Maxdetectors];
        float min_theta = Mathf.PI * 2 / Maxdetectors;

        for (int i = 0; i < Maxdetectors; ++i) {
          Vector3 forward = new Vector3(Mathf.Cos(theta+min_theta*i), 0, Mathf.Sin(theta+min_theta*i)) * detect_dist;
          Vector3 detect_point = transform.position + forward;
          Ray ray = new Ray(transform.position, forward);
          bool isCollider = Physics.Raycast(ray, out RaycastHit hit, forward.magnitude);
          if (isCollider) {
              occupancy_map[i] = (transform.position-hit.point).magnitude / detect_dist;
              Debug.DrawLine(transform.position, detect_point, Color.red);
          }
          else {
              occupancy_map[i] = 1.0f;
              Debug.DrawLine(transform.position, detect_point, Color.green);
          }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // // Heuristic method to test the env
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey("left"))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey("right"))
        {
            discreteActionsOut[0] = 0;
        }
        else discreteActionsOut[0] = 2;
        // var continuousActionsOut = actionsOut.ContinuousActions;
        // continuousActionsOut[0] = Input.GetAxis("Horizontal");
    }

    private Vector3 random_position() {
        float s = 0;
        Vector3 V = new Vector3(0, 0, 0);
        for (int i = 0; i < 4; ++i) {
            float ratio = Random.Range(0.01f, 1f);
            V += vTarget[i].transform.position * ratio;
            s += ratio;
        }
        V = V / s;

        return V;
    }

    private void initializeScene()
    {
        Random.InitState((int)System.DateTime.Now.Ticks);
        index = 0;
        action = 2;

        target = random_position();

        while (true) {
            transform.position = random_position();

            get_occupancy_map();

            int flag = 1;
            for (int i = 0; i < Maxdetectors; ++i)
                if (occupancy_map[i] * detect_dist < 0.1f) {
                    flag = 0;
                    break;
                }
            if (flag == 1)
                break;
        }

        Vector3 pos = transform.position;
        pos.y = 5f;

        transform.position = pos;

        target[1] = transform.position[1];

        change_line(L, transform.position, target);
        
        theta = Random.Range(0, 1f) * Mathf.PI * 2;

        direction = new Vector3(Mathf.Cos(theta), 0, Mathf.Sin(theta));
        transform.LookAt(transform.position+direction);

        distance = (target-transform.position).magnitude / max_distance;

        get_occupancy_map();
    }

    /*
    private bool isColliding(Vector3 obstacle)
    {
        Vector3 head = target.transform.position;
        Vector3 hc = obstacle - head;
        Vector3 hd = transform.position - head;
        
        float angle = Vector3.Angle(hc, hd);

        Debug.DrawRay(head, hc);
        Debug.DrawRay(head, hd);

        return angle < 45f;
    }

    private bool isCollidingRC()
    {
        RaycastHit hit;
        // Bit shift the index of the layer (3) to get a bit mask
        int layerObs = 1 << 3;

        Vector3 dirRC = (target.transform.position - transform.position).normalized;
        if (Physics.Raycast(transform.position, dirRC, out hit, RaycastDist, layerObs))
        {
            Debug.DrawRay(transform.position, dirRC * hit.distance, Color.red);
            // Debug.Log("Did Hit");
            return true;
        }
        else
        {
            Debug.DrawRay(transform.position, dirRC * RaycastDist, Color.green);
            // Debug.Log("Did not Hit");
            return false;
        }
    }
    */
}