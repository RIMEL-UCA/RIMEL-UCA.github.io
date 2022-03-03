import { Sequelize, Model, DataTypes } from "sequelize";
import path from "path";

export const db = new Sequelize({
    dialect: 'sqlite',
    storage: path.resolve(path.resolve(__dirname,'../..'), 'database.sqlite'),
    logging: false,
});