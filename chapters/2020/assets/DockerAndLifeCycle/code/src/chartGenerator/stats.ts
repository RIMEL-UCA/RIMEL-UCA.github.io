export {stats};

import { dictionary } from "./dictionary";
import { toUnicode } from "punycode";

class stats {
    private total: number;
    private totalNotNull: number;
    private expose: number;
    private args: number;
    private volumes: number;
    private EnvVariable: string[];
    private EnvTuple: dictionary[];
    private unknown: string[];
    private unknownTuple: dictionary[];
    private SecurityVariable: string[];
    private securityTuple: dictionary[];
    public metricsList:Array<Metrics> ;
  
    //Constructor
    public constructor() {
      this.total = 0;
      this.totalNotNull = 0;
      this.expose = 0;
      this.args = 0;
      this.volumes = 0;
      this.EnvVariable = [];
      this.unknown = [];
      this.SecurityVariable = [];
      this.EnvTuple = [];
      this.unknownTuple = [];
      this.securityTuple = [];
      this.metricsList = new Array<Metrics>();
      return this;
    };
    
    //Geters 
    public getTotalNotNull() : number{
      return this.totalNotNull;
    };

    public getExpose(): number {
      return this.expose;
    }

    public getArgs(): number {
      return this.args;
    };

    public getVolumes(): number {
      return this.volumes;
    };

    public getEnvVariable() : string[] {
      return this.EnvVariable;
    };

    public getEnvTuple(): dictionary[] {
      return this.EnvTuple;
    };

    public getUnknown(): string[]{
      return this.unknown;
    };
  
    public getUnknownTuple(): dictionary[]{
      return this.unknownTuple;
    };

    public getSecurityVariable(): string[]{
      return this.SecurityVariable;
    };

    public getSecurityTuple(): dictionary[]{
      return this.securityTuple;
    };

    public getTotal() : number {
      return this.total;
    }
    private envToTuple(env: string[]){
      var found = false;
      
      env.forEach(name => {
        this.EnvTuple.forEach(candidate => {
          if(candidate.getName() == name) {
            //varName already appeared
            found = true;
            candidate.addAppareance();
          };
        });
        if (!found) {
          //Add name to tuple
          this.EnvTuple.push(new dictionary(name));
        };
        found = false; //Restart
      });
    };
  
    private unkToTuple(unk: string[]){
      var found = false;
      unk.forEach(name => {
        this.unknownTuple.forEach(candidate => {
          if(candidate.getName() == name) {
            //varName already appeared
            found = true;
            candidate.addAppareance();
          };
        });
        if (!found) {
          //Add name to tuple
          this.unknownTuple.push(new dictionary(name));
        };
        found = false; //Restart
      });
    };
  
    private secToTuple(sec: string[]){
      var found = false;
      sec.forEach(name => {
        this.securityTuple.forEach(candidate => {
          if(candidate.getName() == name) {
            //varName already appeared
            found = true;
            candidate.addAppareance();
          };
        });
        if (!found) {
          //Add name to tuple
          this.securityTuple.push(new dictionary(name));
        };
        found = false; //Restart
      });
    };
    
    public add(metrics: Metrics) {
      this.total++;
      this.totalNotNull++;
      this.expose+=metrics.expose;
      this.args+=metrics.args;
      this.volumes+=metrics.volumes;
      this.EnvVariable = this.EnvVariable.concat(metrics.EnvVariable);
      this.envToTuple(metrics.EnvVariable);
      this.unknown = this.unknown.concat(metrics.unknown);
      this.unkToTuple(metrics.unknown);
      this.SecurityVariable = this.SecurityVariable.concat(metrics.SecurityVariable);
      this.secToTuple(metrics.SecurityVariable);
      this.metricsList.push(metrics);
    };

    
    
    //Add new null
    public addNull() {
      this.total++;
    }
    //Calculated stats
    public exposeAvg() : number{
      if (this.totalNotNull > 0) {
        return this.expose/this.totalNotNull;
      }
      else {
        console.error("No stats found");
      }
    };
    
    public argsAvg() : number{
      if (this.totalNotNull > 0) {
        return this.args/this.totalNotNull;
      }
      else {
        console.error("No stats found");
      }
    };
  
    public volumesAvg() : number{
      if (this.totalNotNull > 0) {
        return this.volumes/this.totalNotNull;
      }
      else {
        console.error("No stats found");
      }
    };

    public envVariablesAvg() : number {
      if (this.totalNotNull > 0) {
        return this.EnvTuple.length/this.totalNotNull;
      }
      else {
        console.error("No stats found");
      }
    };

    public secVariablesAvg() : number {
      if (this.totalNotNull > 0) {
        return this.securityTuple.length/this.totalNotNull;
      }
      else {
        console.error("No stats found");
      }
    };
  }

  export class Metrics {
    expose: number
    args: number
    volumes: number
    EnvVariable: string[] 
    unknown: string[]
    SecurityVariable: string[]
  }
  