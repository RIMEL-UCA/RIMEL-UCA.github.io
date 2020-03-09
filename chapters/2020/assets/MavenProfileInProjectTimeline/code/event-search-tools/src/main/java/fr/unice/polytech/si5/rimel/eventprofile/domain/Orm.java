package fr.unice.polytech.si5.rimel.eventprofile.domain;

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
