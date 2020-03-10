export { dictionary };

class dictionary {
    name: string;
    appareances: number;

    public constructor(nName:string) {
        this.name = nName;
        this.appareances = 1;
    }

    public getName() : string{
        return this.name;
    }

    public getAppareances() : number {
        return this.appareances;
    }

    public addAppareance() {
        this.appareances++;
    }

    public addNAppareances(addNAppareances: number){
        this.appareances += addNAppareances;
    }

    public toString() :  string{
        return "[" + this.name.toString() + ", " + this.appareances.toString() + "]";
    }
}