import { Model, DataTypes } from "sequelize";
import { db } from "./database";
import { Project } from "./project.db";

interface CategorizationAttributes {
    //common
    id: number;
    category: string;
}
export type CategorizationEnum = 'native' | 'express' | 'angular' | 'react' | 'vue' | 'other' | 'nestjs' | 'next';
export class Categorization extends Model<CategorizationAttributes> {
    declare id: number;
    declare owner: string;
    declare name: string;
    declare language: string;
    declare forks: number;
    declare stars: number;
    declare contributors: number;
    declare createdAt: Date;
    declare updatedAt: Date;
}
Categorization.init({
    id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
    },
    category: DataTypes.STRING,
}, { sequelize: db, modelName: 'categorization' });

export function categorizeProject(project: Project, category: CategorizationEnum): Promise<Categorization> {
    return Categorization.create({
        id: project.id,
        category: category,
    });
}

export function clearAllCategorization(): Promise<number> {
    return Categorization.destroy({ where: {} });
}