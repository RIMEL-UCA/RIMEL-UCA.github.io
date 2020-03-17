package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model;

import org.springframework.stereotype.Component;

public enum Orm {
    NONE("none"),
    SPRINGDATA("spring-data"),
    ECLIPSELINK("eclipselink"),
    HIBERNATE("hibernate"),
    OPENJPA("openjpa"),
    SPRINGORM("spring-orm")
    ;
    public final String name;

    Orm(String s) {
        name = s;
    }
}
