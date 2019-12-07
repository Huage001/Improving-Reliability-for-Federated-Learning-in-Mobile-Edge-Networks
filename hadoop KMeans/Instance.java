package KMeans;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;

public class Instance implements Writable{
    private ArrayList<Double> value;

    public Instance(){
        value = new ArrayList<Double>();
    }

    public Instance(String s){
        String[] valuestr = s.split("[, ]");
        value = new ArrayList<Double>();
        for(String val:valuestr){
            value.add(Double.parseDouble(val));
        }
    }

    public Instance(Instance ins){
        value = new ArrayList<Double>();
        for(double val:ins.getValue()){
            value.add(val);
        }
    }

    public ArrayList<Double> getValue(){
        return value;
    }

    public String toString(){
        String str = "";
        for(int i = 0; i < value.size(); i++){
            if(i != value.size()-1)  str = str.concat(Double.toString(value.get(i)) + ",");
            else str = str.concat(Double.toString(value.get(i)));
        }
        return str;
    }

    public Instance add(Instance ins){
        if(value.size() != ins.getValue().size()) System.out.printf("向量维数不同，不可相加！\n");
        Instance result = new Instance();
        for(int i = 0; i < value.size(); i++){
            result.value.add(value.get(i) + ins.getValue().get(i));
        }
        return result;
    }
    public Instance multiply(long n){
        Instance result = new Instance(this);
        for(int i = 0; i < value.size(); i++){
            result.value.set(i, value.get(i)*n);
        }
        return result;
    }
    public Instance divide(long n){
        Instance result = new Instance(this);
        for(int i = 0; i < value.size(); i++){
            result.value.set(i, value.get(i)/n);
        }
        return result;
    }

    public double calDisSq(Instance ins){
        if(value.size() != ins.getValue().size()) System.out.printf("向量维数不同，不可计算距离！\n");
        double dis = 0;
        for(int i = 0; i < value.size(); i++){
            dis += (value.get(i) - ins.getValue().get(i)) * (value.get(i) - ins.getValue().get(i));
        }
        return dis;
    }
    public int chooseNearestCluster(ArrayList<Cluster> kClusters){
        double minDis = Double.MAX_VALUE;
        int nearestClusterID = -1;
        double dis = 0;
        for(Cluster c:kClusters){
            dis = calDisSq(c.getCenter());
            if(dis < minDis){
                minDis = dis;
                nearestClusterID = c.getClusterID();
            }
        }
        return nearestClusterID;
    }

    @Override
    public void write(DataOutput out) throws IOException{
        out.writeInt(value.size());
        for(int i = 0; i < value.size(); i++){
            out.writeDouble(value.get(i));
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException{
        value = new ArrayList<Double>();
        int size = in.readInt();
        if(size != 0){
            for(int i = 0; i < size; i++){
                value.add(in.readDouble());
            }
        }
    }
}
